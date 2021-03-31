import torch
import torch.nn as nn
import sys


class EMB(nn.Module):
    def __init__(self, half_channel):
        super(EMB, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(half_channel, half_channel, kernel_size=1),
                nn.BatchNorm2d(half_channel),
                nn.ReLU(True),
                nn.Conv2d(half_channel, half_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(half_channel),
                nn.ReLU(True),
                nn.Conv2d(half_channel, half_channel, kernel_size=1),
                nn.BatchNorm2d(half_channel),
                nn.ReLU(True)
        )
        
    def forward(self, features):
        if not self.training:
            half_features = torch.chunk(features, 2, dim=1)
            #Produce enhanced map
            spatial_attn = torch.mean(features, dim=1, keepdim=True)
            importance_map = torch.sigmoid(spatial_attn)
            concat_feature_map = self._features_concat(half_features)
            return (importance_map * concat_feature_map) + features
        else:
            half_features = torch.chunk(features, 2, dim=1)
            # Produce enhanced map
            spatial_attn = torch.mean(features, dim=1, keepdim=True)
            importance_map = torch.sigmoid(spatial_attn)
            concat_feature_map = self._features_concat(half_features)
            return (importance_map * concat_feature_map) + features

    def _features_concat(self, half_features):
        conv_half_features = self.conv(half_features[1])
        return torch.cat((half_features[0], conv_half_features), dim=1)