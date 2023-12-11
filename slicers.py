import torch
import torch.nn as nn
# from utils import one_dimensional_Wasserstein

class Base_Slicer(nn.Module):
    def __init__(self,d,L):
        super(Base_Slicer, self).__init__()
        self.d=d
        self.L=L
        self.U = nn.Linear(self.d,self.L,bias=False)
        self.U.weight.data = torch.randn(self.U.weight.shape,device='cuda')
        self.project_parameters()
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        self.project_parameters()
        return self.U(x)
    def project_parameters(self):
        self.U.weight.data = self.U.weight/torch.sqrt(torch.sum(self.U.weight**2,dim=1,keepdim=True))
    def reset(self):
        self.U.weight.data = torch.randn(self.U.weight.shape,device='cuda')
        self.project_parameters()


class NonLinearBase_Slicer(nn.Module):
    def __init__(self,d,L,activation=nn.Sigmoid()):
        super(NonLinearBase_Slicer, self).__init__()
        self.activation =activation
        self.d=d
        self.L=L
        self.U = nn.Linear(self.d,self.L,bias=False)
        self.U.weight.data = torch.randn(self.U.weight.shape,device='cuda')
        self.project_parameters()
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        self.project_parameters()
        x = self.U(x)
        x = self.activation(x)
        return x
    
    def project_parameters(self):
        self.U.weight.data = self.U.weight/torch.sqrt(torch.sum(self.U.weight**2,dim=1,keepdim=True))
    def reset(self):
        self.U.weight.data = torch.randn(self.U.weight.shape,device='cuda')
        self.project_parameters()


class Conv_MNIST_Slicer(nn.Module):
    def __init__(self,L):
        super(Conv_MNIST_Slicer, self).__init__()
        image_sizes=(1,28,28)
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=4, stride=2, padding=1, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=4, stride=2, padding=1, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape,device='cuda')
        self.U2.weight.data = torch.randn(self.U2.weight.shape,device='cuda')
        self.U3.weight.data = torch.randn(self.U3.weight.shape,device='cuda')
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.U3(self.U2(self.U1(x)))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape,device='cuda')
        self.U2.weight.data = torch.randn(self.U2.weight.shape,device='cuda')
        self.U3.weight.data = torch.randn(self.U3.weight.shape,device='cuda')
        self.project_parameters()

class NonLinearConv_MNIST_Slicer(nn.Module):
    def __init__(self,L, activation=nn.Sigmoid()):
        super(NonLinearConv_MNIST_Slicer, self).__init__()
        image_sizes=(1,28,28)
        self.activation=activation
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=4, stride=2, padding=1, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=4, stride=2, padding=1, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape,device='cuda')
        self.U2.weight.data = torch.randn(self.U2.weight.shape,device='cuda')
        self.U3.weight.data = torch.randn(self.U3.weight.shape,device='cuda')
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        return self.activation(self.U3(self.activation(self.U2(self.activation(self.U1(x))))))

    def project_parameters(self):
        self.U1.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U2.weight.data = self.U2.weight / torch.sqrt(torch.sum(self.U2.weight ** 2,dim=[1,2,3],keepdim=True))
        self.U3.weight.data = self.U3.weight / torch.sqrt(torch.sum(self.U3.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        self.U1.weight.data = torch.randn(self.U1.weight.shape,device='cuda')
        self.U2.weight.data = torch.randn(self.U2.weight.shape,device='cuda')
        self.U3.weight.data = torch.randn(self.U3.weight.shape,device='cuda')
        self.project_parameters()

class NonLinearConv_MNIST_Slicer_(nn.Module):
    def __init__(self,L, activation=[None, None, None]):
        super(NonLinearConv_MNIST_Slicer_, self).__init__()
        image_sizes=(1,28,28)
        self.activation=activation
        self.U1 = nn.Conv2d(image_sizes[0], 1*L, kernel_size=4, stride=2, padding=1, bias=False)
        self.U2 = nn.Conv2d(1*L, 1 * L, kernel_size=4, stride=2, padding=1, bias=False,groups=L)
        self.U3 = nn.Conv2d(1*L, 1 * L, kernel_size=7, stride=1, padding=0, bias=False, groups=L)
        self.U1.weight.data = torch.randn(self.U1.weight.shape,device='cuda')
        self.U2.weight.data = torch.randn(self.U2.weight.shape,device='cuda')
        self.U3.weight.data = torch.randn(self.U3.weight.shape,device='cuda')
        self.U_list = [self.U1, self.U2, self.U3]
        self.project_parameters()

    def forward(self, x):
        self.project_parameters()
        for i, U in enumerate(self.U_list):
            if self.activation[i]:
                x=self.activation[i](U(x))
            else:
                x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = self.U1.weight / torch.sqrt(torch.sum(self.U1.weight ** 2,dim=[1,2,3],keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape,device='cuda')
        self.project_parameters()


class ConvSlicer(nn.Module):
    def __init__(self,L,ch=3,bottom_width=8,type='csw'):
        super(ConvSlicer, self).__init__()
        if (bottom_width == 32):
            self.U1 = nn.Conv2d(ch, L, kernel_size=17, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5]
        elif (bottom_width == 64):
            self.U1 = nn.Conv2d(ch, L, kernel_size=33, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=17, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U6 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5, self.U6]
        elif (bottom_width == 8):
            self.U1 = nn.Conv2d(ch, L, kernel_size=5, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3]
        self.reset()

    def forward(self, x):
        self.project_parameters()
        for U in self.U_list:
            x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2,dim=[1,2,3],keepdim=True))
    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape,device='cuda')
        self.project_parameters()


class NonLinearConvSlicer(nn.Module):
    def __init__(self,L,ch=3,bottom_width=8,type='ncsw',activation=nn.Sigmoid()):
        super(NonLinearConvSlicer, self).__init__()
        self.activation=activation
        if (bottom_width == 32):
            self.U1 = nn.Conv2d(ch, L, kernel_size=17, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5]
        elif (bottom_width == 64):
            self.U1 = nn.Conv2d(ch, L, kernel_size=33, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=17, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U6 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5, self.U6]
        elif (bottom_width == 8):
            self.U1 = nn.Conv2d(ch, L, kernel_size=5, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3]
        self.reset()

    def forward(self, x):
        self.project_parameters()
        for i in range(len(self.U_list)-1):
            x = self.activation(self.U_list[i](x))
        x=self.U_list[-1](x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
        self.project_parameters()


class NonLinearConvSlicer_(nn.Module):
    def __init__(self,L,ch=3,bottom_width=8,type='ncsw',activation=[None, None, None, None, None]):
        super(NonLinearConvSlicer_, self).__init__()
        self.activation=activation
        if (bottom_width == 32):
            self.U1 = nn.Conv2d(ch, L, kernel_size=17, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5]
        elif (bottom_width == 64):
            self.U1 = nn.Conv2d(ch, L, kernel_size=33, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=17, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=9, stride=1, padding=0, bias=False, groups=L)
            self.U4 = nn.Conv2d(L, L, kernel_size=5, stride=1, padding=0, bias=False, groups=L)
            self.U5 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U6 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5, self.U6]
        elif (bottom_width == 8):
            self.U1 = nn.Conv2d(ch, L, kernel_size=5, stride=1, padding=0, bias=False)
            self.U2 = nn.Conv2d(L, L, kernel_size=3, stride=1, padding=0, bias=False, groups=L)
            self.U3 = nn.Conv2d(L, L, kernel_size=2, stride=1, padding=0, bias=False, groups=L)
            self.U_list = [self.U1, self.U2, self.U3]
        self.reset()

    def forward(self, x):
        self.project_parameters() #bo cai nay di
        for i, U in enumerate(self.U_list):
            if self.activation[i]:
                x=self.activation[i](U(x))
            else:
                x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
        self.project_parameters()

class Hierarchical_Slicer(nn.Module):
    def __init__(self, d, L, f_dim, device='cuda', activation=None):
        super(Hierarchical_Slicer, self).__init__()
        self.d = d
        self.L = L
        self.f_dim = f_dim
        self.device = device
        self.U1 = nn.Conv2d(self.d, 64, 4, 2, 1, bias=False)
        self.U2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.U3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.U4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.U5 = nn.Linear(self.f_dim,self.L,bias=False)
        self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5]
        if activation:
            self.activation = activation
        else:
            self.activation = [None]*(len(self.U_list))
        self.reset()

    def forward(self, x):
        # self.project_parameters() #bo cai nay di
        for i, U in enumerate(self.U_list):
            if i == len(self.U_list) - 1:
                x = x.view(x.size(0), -1)
            if self.activation[i]:
                x=self.activation[i](U(x))
            else:
                x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list[:-1]:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))
        self.U_list[-1].weight.data = self.U_list[-1].weight/torch.sqrt(torch.sum(self.U_list[-1].weight**2,dim=1,keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
            # U.weight.data = torch.randn(U.weight.shape)
        self.project_parameters()

class Hierarchical_Slicer_MNIST(nn.Module):
    def __init__(self, L, device='cuda', activation=None):
        super(Hierarchical_Slicer_MNIST, self).__init__()
        self.L = L
        self.device = device
        self.U1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.U2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.U3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.U4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.U5 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False)
        self.U6 = nn.Linear(256 * 5 * 5,self.L,bias=False)
        self.U_list = [self.U1, self.U2, self.U3, self.U4, self.U5, self.U6]
        for u in self.U_list:
            print(u)
        if activation:
            self.activation = activation
        else:
            self.activation = [None]*(len(self.U_list)+1)
        # print(self.activation)
        self.reset()

    def forward(self, x):
        # self.project_parameters() #bo cai nay di
        for i, U in enumerate(self.U_list):
            if i == len(self.U_list) - 1:
                x = x.view(x.size(0), -1)
            if self.activation[i]:
                x=self.activation[i](U(x))
            else:
                x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list[:-1]:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))
        self.U_list[-1].weight.data = self.U_list[-1].weight/torch.sqrt(torch.sum(self.U_list[-1].weight**2,dim=1,keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
            # U.weight.data = torch.randn(U.weight.shape)
        self.project_parameters()



class GSW(nn.Module):
    def __init__(self,ftype='linear',d=28*28,L=10,degree=2,radius=2.,use_cuda=True):
        super(GSW, self).__init__()
        self.ftype=ftype
        self.nofprojections=L
        self.d=d
        self.degree=degree
        self.radius=radius
        if torch.cuda.is_available() and use_cuda:
            self.device=torch.device('cuda')
        else:
            self.device=torch.device('cpu')
        self.theta=None # This is for max-GSW


    def forward(self,x):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        x = x.view(x.shape[0],-1)
        if self.ftype=='linear':
            return self.linear(x,self.theta)
        elif self.ftype=='poly':
            return self.poly(x,self.theta)
        elif self.ftype=='circular':
            return self.circular(x,self.theta)
        else:
            raise Exception('Defining function not implemented')

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())

    def poly(self,X,theta):
        ''' The polynomial defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
            degree: degree of the polynomial
        '''
        N,d=X.shape
        assert theta.shape[1]==self.homopoly(d,self.degree)
        powers=list(self.get_powers(d,self.degree))
        HX=torch.ones((N,len(powers))).to(self.device)
        for k,power in enumerate(powers):
            for i,p in enumerate(power):
                HX[:,k]*=X[:,i]**p
        if len(theta.shape)==1:
            return torch.matmul(HX,theta)
        else:
            return torch.matmul(HX,theta.t())

    def circular(self,X,theta):
        ''' The circular defining function for generalized Radon transform
            Inputs
            X:  Nxd matrix of N data samples
            theta: Lxd vector that parameterizes for L projections
        '''
        N,d=X.shape
        if len(theta.shape)==1:
            return torch.sqrt(torch.sum((X-theta)**2,dim=1))
        else:
            return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)

    def get_powers(self,dim,degree):
        '''
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        '''
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1,degree - value):
                    yield (value,) + permutation

    def homopoly(self,dim,degree):
        '''
        calculates the number of elements in a homogeneous polynomial
        '''
        return len(list(self.get_powers(dim,degree)))
    

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='poly':
            dpoly=self.homopoly(dim,self.degree)
            theta=torch.randn((self.nofprojections,dpoly))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype=='circular':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([self.radius*th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)
        # return theta
    
    def reset(self):
        self.theta=self.random_slice(self.d)

class SH_Slicer(nn.Module):
    def __init__(self, L, device='cuda', activation=None):
        super(SH_Slicer, self).__init__()
        self.L = L
        self.device = device
        self.U1 = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.U2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.U3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.U4 = nn.Conv2d(256, self.L, kernel_size=(4, 4), stride=(1, 1), bias=False)
        self.U_list = [self.U1, self.U2, self.U3, self.U4]
        if activation:
            self.activation = activation
        else:
            self.activation = [None]*(len(self.U_list))
        self.reset()

    def forward(self, x):
        # self.project_parameters() #bo cai nay di
        for i, U in enumerate(self.U_list):
            if self.activation[i]:
                x=self.activation[i](U(x))
            else:
                x=U(x)
        return x

    def project_parameters(self):
        for U in self.U_list:
            U.weight.data = U.weight / torch.sqrt(torch.sum(U.weight ** 2, dim=[1, 2, 3], keepdim=True))

    def reset(self):
        for U in self.U_list:
            U.weight.data = torch.randn(U.weight.shape, device='cuda')
            # U.weight.data = torch.randn(U.weight.shape)
        self.project_parameters()

if __name__=="__main__":
    # x = torch.rand([128, 3, 32, 32])
    # theta = Hierarchical_Slicer(3, [64, 128, 256, 512], 100, 512*2*2)
    # for u in theta.U_list:
    #     print(u)
    # x_projected = theta(x)
    # print(x_projected.shape)

    # x = torch.rand([128, 1, 28, 28])
    # theta = Hierarchical_Slicer_MNIST(100)
    # for u in theta.U_list:
    #     print(u)
    # x_projected = theta(x)
    # print(x_projected.shape)
    import random
    random.seed(0)
    torch.manual_seed(0)

    # x = torch.rand([64, 1, 28, 28])
    # print(x)
    # # slicer = GSW(ftype='poly', d=28*28, L =1, degree = 3)
    # slicer = GSW(ftype='circular', d=28*28, L =1000)
    # slicer.reset()
    # x_projected = slicer(x)
    # print(x_projected)
    