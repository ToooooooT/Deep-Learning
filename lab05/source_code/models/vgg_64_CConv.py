import torch
import torch.nn as nn

class vgg_layer(nn.Module):
    def __init__(self, nin, nout, cond_dim):
        super(vgg_layer, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 3, 1, 1)
        self.fc1 = nn.Linear(cond_dim, nout)
        self.fc2 = nn.Linear(cond_dim, nout)

        self.main = nn.Sequential(
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
        )
        

    def forward(self, input, cond):
        x = self.conv(input)
        s = nn.functional.softplus(self.fc1(cond)).unsqueeze(-1).unsqueeze(-1)
        b = self.fc2(cond).unsqueeze(-1).unsqueeze(-1)
        y = x * s + b
        return self.main(y)


class vgg_encoder(nn.Module):
    def __init__(self, dim, cond_dim):
        super(vgg_encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = vgg_layer(3, 64, cond_dim)
        self.c2 = vgg_layer(64, 64, cond_dim)
        # 32 x 32
        self.c3 = vgg_layer(64, 128, cond_dim)
        self.c4 = vgg_layer(128, 128, cond_dim)
        # 16 x 16 
        self.c5 = vgg_layer(128, 256, cond_dim)
        self.c6 = vgg_layer(256, 256, cond_dim)
        self.c7 = vgg_layer(256, 256, cond_dim)
        # 8 x 8
        self.c8 = vgg_layer(256, 512, cond_dim)
        self.c9 = vgg_layer(512, 512, cond_dim)
        self.c10 = vgg_layer(512, 512, cond_dim)
        # 4 x 4
        self.c11 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input, cond):
        h = self.c1(input, cond)
        h1 = self.c2(h, cond)
        h = self.c3(self.mp(h1), cond)
        h2 = self.c4(h, cond)
        h = self.c5(self.mp(h2), cond)
        h = self.c6(h, cond)
        h3 = self.c7(h, cond)
        h = self.c8(self.mp(h3), cond)
        h = self.c9(h, cond)
        h4 = self.c10(h, cond)
        h5 = self.c11(self.mp(h4))
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class vgg_decoder(nn.Module):
    def __init__(self, dim, cond_dim):
        super(vgg_decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = vgg_layer(512*2, 512, cond_dim)
        self.upc3 = vgg_layer(512, 512, cond_dim)
        self.upc4 = vgg_layer(512, 256, cond_dim)
        # 16 x 16
        self.upc5 = vgg_layer(256*2, 256, cond_dim)
        self.upc6 = vgg_layer(256, 256, cond_dim)
        self.upc7 = vgg_layer(256, 128, cond_dim)
        # 32 x 32
        self.upc8 = vgg_layer(128*2, 128, cond_dim)
        self.upc9 = vgg_layer(128, 64, cond_dim)
        # 64 x 64
        self.upc10 = vgg_layer(64*2, 64, cond_dim)
        self.upc11 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input, cond):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        up1 = self.up(d1)
        up = self.upc2(torch.cat([up1, skip[3]], 1), cond)
        up = self.upc3(up, cond)
        d2 = self.upc4(up, cond)
        up2 = self.up(d2)
        up = self.upc5(torch.cat([up2, skip[2]], 1), cond)
        up = self.upc6(up, cond)
        d3 = self.upc7(up, cond)
        up3 = self.up(d3)
        up = self.upc8(torch.cat([up3, skip[1]], 1), cond)
        d4 = self.upc9(up, cond)
        up4 = self.up(d4)
        up = self.upc10(torch.cat([up4, skip[0]], 1), cond)
        output = self.upc11(up)
        return output

