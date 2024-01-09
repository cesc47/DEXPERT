import torch
from torch import nn
import os


def adapt_model(args, model, num_classes):
    """
    Adapt the model to the number of classes and the type of the task (classification or regression). add segmentation
    maps to the input if needed
    Args:
        args:  command line arguments
        model:  model to adapt
        num_classes:  number of classes
    Returns:
        adapted model
    """
    if args.model.startswith('vit'):
        if args.use_segmentation_maps:
            model.conv_proj = torch.nn.Conv2d(4, model.hidden_dim, kernel_size=16, stride=16)  # change input ch. to 4
        model.heads.head = torch.nn.Linear(model.hidden_dim, num_classes, bias=True)  # 1 class => regression
        if num_classes == 1:
            model.heads.head.bias.data = torch.tensor([1975.0])
    elif args.model.startswith('conv'):
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, num_classes, bias=True)
        )
        if num_classes == 1:  # if regression is performed, initialize the bias to 1975 to stabilize the training
            model.classifier[1].bias.data = torch.tensor([1975.0])
    elif args.model.startswith('ResNet'):
        model.fc = torch.nn.Linear(2048, num_classes, bias=True)
    else:
        raise NotImplementedError('Only ViT and ConvNext models are supported for training DEW dataset')

    return model


class DEXPERTS(nn.Module):
    def __init__(self, backbone):
        """
        Args: command line arguments
        backbone: adapted model (convnext in this case).
        """
        super(DEXPERTS, self).__init__()
        self.specialists = ['general', 'person', 'car', 'boat', 'bus', 'airplane', 'train', 'padded']
        self.num_classes = 14
        self.d_ff = 512
        self.heads = 4
        self.depth = 2
        self.num_cnn_features = len(self.specialists)
        self.dim_cnn_features = 1024  # output from Convnext
        self.dim_transformer = 256

        # load specialists
        for model_name in os.listdir('../models'):
            specialist_name = model_name.split('.')[0].split('-')[-1]  # split the last part of the path (model name)
            # take backbone and load the weights of the model.
            self.__setattr__(specialist_name, backbone)
            checkpoint = torch.load(os.path.join('../models', model_name), map_location="cpu")
            self.__getattr__(specialist_name).load_state_dict(checkpoint["model"], strict=False)
            for param in self.__getattr__(specialist_name).parameters():  # freeze the weights of the specialist
                param.requires_grad = False
            # remove the classifier of the specialist.
            self.__getattr__(specialist_name).classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
            )

        # ----------------------------------- BUILD TRANSFORMER -----------------------------------
        # Add a linear projection of cnn features to the transformer dim
        self.cnn_feature_to_embedding = torch.nn.Linear(self.dim_cnn_features, self.dim_transformer)
        # position embedding for the transformer. The positional embedding is just something that tells the transformer
        # that not all tokens are the same thing. e.g. the 1st tokes is global visual features, the 2nd token are
        # "person" features, the 3rd are "car" features, etc...
        self.pos_embedding = nn.Embedding(self.num_cnn_features, self.dim_transformer)
        # transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_transformer, nhead=self.heads, dim_feedforward=self.d_ff, batch_first=True)
        # transformer encoder (stack of encoder layers)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)
        # fc to num_classes
        self.fc = nn.Linear(self.dim_transformer, self.num_classes)
        # -----------------------------------------------------------------------------------------

    def forward(self, x):
        # output(specialist) => linear projection => transformer => cls token => MLP head
        # output (x[0]) is of size (batch, seq_len, rgb, h, w)
        # in x[1] we have the specialist names
        specialist_names = x[1]

        # mask to remove the padded tokens: False if the token is not padded, True otherwise. specialists_names variable
        # contains the name of the specialist for each token. If the name is "padded", the token is padded. a tensor of
        # shape (batch_size, seq_len) has to be created
        mask = torch.zeros((x[0].shape[0], len(x[1][0])), device=x[0].device)
        for batch_dim in range(x[0].shape[0]):
            for idx, specialist in enumerate(specialist_names[batch_dim]):
                if specialist == 'padded':
                    mask[batch_dim, idx] = 1

        # create an empty tensor to store the output of the specialists.
        # It has to be of shape (batch_size, seq_len, dim_cnn_features)
        outputs = torch.zeros((x[0].shape[0], len(x[1][0]), self.dim_cnn_features), device=x[0].device)

        for batch_dim in range(x[0].shape[0]):
            # iterate over the x[1] (specialists)
            for idx, specialist in enumerate(specialist_names[batch_dim]):
                if specialist != 'padded':
                    # forward through the specialist
                    outputs[batch_dim, idx, :] = self.__getattr__(specialist)(x[0][batch_dim, idx, :, :, :].unsqueeze(0))

        # forward through the linear projection
        x = self.cnn_feature_to_embedding(outputs)
        # for each element of the batch, get the position embedding: the position in self.specialists.
        # add the position embedding to the output of the linear projection
        for batch_dim in range(x.shape[0]):
            x[batch_dim, :, :] += self.pos_embedding(torch.tensor([self.specialists.index(specialist) for specialist in specialist_names[batch_dim]], device=x.device))
        # forward through the transformer
        x = self.encoder(x, src_key_padding_mask=mask.bool())
        # here the output is of shape (batch_size, self.num_cnn_features, 14). We want to average over the specialists
        # dimension, so that we have a tensor of shape (batch_size, 14).
        x = torch.mean(x, dim=1)
        # forward through the fc
        x = self.fc(x)

        return x
