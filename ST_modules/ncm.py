import torch
import torch.nn as nn


class NCM_classifier(nn.Module):
    def __init__(self, features, num_classes, deep_nno=False):
        super(NCM_classifier, self).__init__()
        self.means = nn.Parameter(torch.zeros(num_classes, features), requires_grad=False)
        self.mean_distance = nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        self.running_means = nn.Parameter(torch.zeros(num_classes, features), requires_grad=False)
        self.counter_means = nn.Parameter(torch.zeros(num_classes), requires_grad=False)##.cuda()
        self.features = features
        self.classes = num_classes
        self.variance = nn.Parameter(torch.zeros(1)-1, requires_grad=False)
        self.use_variance = False
        self.deep_nno = deep_nno

    def reject(self, x, dist, tau=None):
        out = torch.zeros(x.shape[0], x.shape[1] + 1).to(x.device)

        if self.deep_nno or self.nno:
            out[:, :x.shape[1]] = (x > tau).float() * 1. * x
        else:
            out[:, :x.shape[1]] = (dist <= tau).float() * 1. * x

        # last column contains probabilities (distances) for unknown.
        out[:, -1] = 1. - ((out[:, :x.shape[1]]).sum(1) > 0).float()
        return out


    def forward(self, x):
        if self.deep_nno:
            means_reshaped = self.means.view(1, self.classes, self.features).expand(x.data.shape[0],
                                                                                    self.classes,
                                                                                    self.features)###
            features_reshaped = x.view(-1, 1, self.features).expand(x.data.shape[0], self.classes, self.features)###

            #print(f'MEANS: {self.means} \n MEANS_RESHAP: {means_reshaped}')
            #print(f'FEATURES: {x} \n FEATURES_RESHAP: {features_reshaped}')

            diff = (features_reshaped - means_reshaped) ** 2
            cumulative_diff = diff.sum(dim=-1)
            exponent = -cumulative_diff / 2.
            return torch.exp(exponent), exponent, cumulative_diff
        else:
            if self.use_variance:
                std = x.std()
            else:
                std = self.variance
            x = x / std

            means_reshaped = self.means.view(1, self.classes, self.features).expand(x.data.shape[0],
                                                                                    self.classes, self.features)

            features_reshaped = x.view(-1, 1, self.features).expand(x.data.shape[0], self.classes, self.features)

            cumulative_diff = torch.norm((features_reshaped - means_reshaped), dim=2, p=2)
            exponent = - cumulative_diff
            return torch.softmax(exponent, dim=1), exponent, cumulative_diff

    def forward_nno(self, x):

        means_reshaped = self.means.view(1, self.classes, self.features).expand(x.data.shape[0], self.classes,
                                                                                self.features)
        features_reshaped = x.view(-1, 1, self.features).expand(x.data.shape[0], self.classes, self.features)

        cumulative_diff = (features_reshaped - means_reshaped).pow(2).sum(dim=-1)
        exponent = - cumulative_diff / 2.
        return torch.exp(exponent), exponent, cumulative_diff

    # Compute mean by filtering the data of the same label
    @staticmethod
    def compute_mean(x, y, i, weight):
        mask = (y == i)
        #print(mask,y, type(y))
        N = mask.sum()  # num of imgs of class i in the batch
        if weight is not None:
            mask = mask * weight
            mask = mask / mask.sum() * N
        if N == 0:
            return N, 0
        else:
            return N, x[mask].sum(axis=0)

    @staticmethod
    def compute_dist(x, mean):
        return torch.norm(x - mean, p=2, dim=1)

    # Update centers (x=features, y=labels)
    def update_means(self, x, y, std=True, weight=None, alpha=0.8):
        if not self.deep_nno:
            if self.variance.data == -1:
                self.variance.data = x.data.std()
            else:
                self.variance.data = alpha * self.variance + (1 - alpha) * x.data.std()

            if std:
                x = x / x.std()

        for i in range(self.classes):
            # For each label
            N, mean = self.compute_mean(x, y, i+1, weight)  # Compute mean
            # If labels already in the set, just update holder, otherwise add it to the model
            if N == 0:
                continue
            else:
                # if there is not a mean for that class
                if self.counter_means[i] == 0:
                    # add the mean
                    #print(f'ACRESCENTA NO NCM A MÉDIA DA CLASSE {i}')
                    self.means.data[i, :] = mean / N
                    # for each sample of the batch compute the distance from the mean of the i-th class
                    dist = self.compute_dist(x, mean)
                    # select from the batch only the sample of class i. For them, average the distances they have from
                    # the mean of their class. Here there is the sum since there are no previous stored distances
                    self.mean_distance.data[i] = dist[y == i+1].sum()

                else:
                    # if there already is a mean for that class, update it with the new batch
                    #print(f'ATUALIZA NO NCM A MÉDIA DA CLASSE {i}')
                    self.means.data[i, :] = (self.means.data[i, :] * alpha + mean * (1-alpha) / N)
                    dist = self.compute_dist(x, self.means.data[i, :])
                    mean_dist = dist[y == i+1].mean()
                    self.mean_distance.data[i] = self.mean_distance.data[i] * alpha + mean_dist * (1-alpha)
                self.counter_means[i] += N # qtd de objs de treino relativo a classe i

    def get_average_dist(self, dim=-1):
        if dim == 0:
            return self.mean_distance.mean()
        return self.mean_distance

    def reset(self):
        self.counter_means = nn.Parameter(torch.zeros(self.classes), requires_grad=False)#.cuda()

    def add_classes(self, new_classes):
        new_class_means = torch.zeros(new_classes, self.features)#.cuda()
        new_class_counters = torch.zeros(new_classes)#.cuda()  # Class Means
        new_running_means = torch.zeros(new_classes, self.features)#.cuda()
        new_mean_dist = torch.zeros(new_classes)#.cuda()
        self.means = nn.Parameter(torch.cat((self.means.data, new_class_means), 0), requires_grad=False)
        self.running_means = nn.Parameter(torch.cat((self.running_means.data, new_running_means), 0),
                                          requires_grad=False)
        self.counter_means = nn.Parameter(torch.cat((self.counter_means, new_class_counters), 0),requires_grad=False)
        self.mean_distance = nn.Parameter(torch.cat((self.mean_distance, new_mean_dist), 0), requires_grad=False)

        print(f"Adding new classes to NCM: from {self.classes} to {self.classes + new_classes}")
        self.classes += new_classes


    def eval(self):
        self.use_variance = False

    def train(self, mode=True):
        self.use_variance = mode
