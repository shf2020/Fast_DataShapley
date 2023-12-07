import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from utils import ShapleySampler, DatasetRepeat
from tqdm.auto import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
import copy

def additive_efficient_normalization(pred, grand, null):
    '''
    Apply additive efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    '''
    # print(grand.size(),pred.shape)
    gap = (grand - null) - torch.sum(pred, dim=1)
    # print(gap.size())
    # gap = gap.detach()
    # print(gap.unsqueeze(1).size(),pred.shape[1])
    return pred + gap.unsqueeze(1) / pred.shape[1]


def evaluate_explainer(explainer, normalization, x, grand, null, num_players):
    '''
    Helper function for evaluating the explainer model and performing necessary
    normalization and reshaping operations.

    Args:
      explainer: explainer model.
      normalization: normalization function.
      x: input.
      grand: grand coalition value.
      null: null coalition value.
      num_players: number of players.
      inference: whether this is inference time (or training).
    '''
    # Evaluate explainer.
    pred = explainer(x)
    # print("1------------------")
    # print(pred.shape,len(x)) #torch.Size([len(x), 1000, 10])
    # Reshape SHAP values.
    if len(pred.shape) == 4: 
        pred = pred.reshape(len(x),  num_players, -1)
  
    else:
        pred = pred.reshape(len(x), num_players, -1)
    
    # For pre-normalization efficiency gap.
    total = pred.sum(dim=1)
    # Apply normalization.
    if normalization:
        # print(pred.shape)
        pred = normalization(pred, grand, null)
     
    return pred, total


def generate_validation_data(X_raw, y_raw, val_set, label_list, null_model, stop_epoch, sampler,
                              link,  class_lr):
    '''
    Generate coalition values for validation dataset.

    Args:
      val_set: validation dataset object.
      validation_samples: number of samples per validation example.
      sampler: Shapley sampler.
      batch_size: minibatch size.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    '''
    # Generate coalitions.
    # Get values.
    val_values = []
    val_S = []
    for x in val_set:
        # Sample S.
        hat_alliance = label_list
        S = []
        hat_S = sampler.sample(1, paired_sampling=True)
        hat_S_nozero = hat_S[0].nonzero().reshape(-1)
        for id in hat_S_nozero:
            if isinstance(hat_alliance[id],list):   
                S.extend(hat_alliance[id])
            else:
                S.append(hat_alliance[id])
        # print(hat_alliance)      
        # print(len(S)," hat_S_nozero:", hat_S_nozero,"S:",S)
        val_S.append(hat_S)
        # print("Classification model_val training data size",torch.tensor(X_raw)[S_nozero].size())
        X_S = torch.tensor(X_raw)[S]
        y_X = torch.tensor(y_raw)[S]

        train_dataset = TensorDataset(X_S,y_X)
        class_train_loader=DataLoader(train_dataset, batch_size=20, shuffle=True)
        
        net = copy.deepcopy(null_model).cuda()
        class_optimizer=torch.optim.Adam(net.parameters(),lr=class_lr)
        loss_function=nn.CrossEntropyLoss()
        values = 0
        for ep in range(stop_epoch):
        # 记录把所有数据集训练+测试一遍需要多长时间
            for img, label in class_train_loader:  # 对于训练集的每一个batch
                # print(img.shape,label)  
                img = img.cuda()
                label = label.cuda()
                out = net( img )  # 送进网络进行输出
                loss = loss_function( out, label ) 
                class_optimizer.zero_grad()
                loss.backward()
                class_optimizer.step()
            # print("训练第{}个epoch".format(ep))
            # print(x[0].shape,values)
            with torch.no_grad():
                values += link(net(x[0].view(1,1,28,28).cuda()))
            # print(x[0].shape,values)
        values /= stop_epoch   
        val_values.append(values.cpu())
        # 释放模型内存
        del net
        # 清理GPU缓存
        torch.cuda.synchronize()
    # val_S = torch.stack(val_S, dim=0)
    val_values = torch.stack(val_values, dim=0)
    
    return val_S, val_values


def validate(val_loader, val_S, label_list, num_players, explainer, grand_model, null_model, link, normalization):
    '''
    Calculate mean validation loss.

    Args:
      val_loader: validation data loader.
      explainer: explainer model.
      null: null coalition value.
      link: link function.
      normalization: normalization function.
    '''
    with torch.no_grad():
        # Setup.
        device = next(explainer.parameters()).device
        mean_loss = 0
        N = 0
        loss_fn = nn.MSELoss()
        # print("1--------------")
        for i,(x,y,v) in enumerate(val_loader):
            # print("2--------------")
            # Move to device.
            x = x.cuda()
            S = []
            hat_S = val_S[i].cuda()
            hat_S_nozero = hat_S[0].nonzero().reshape(-1)
            hat_alliance = label_list
            for id in hat_S_nozero:
                if isinstance(hat_alliance[id],list):   
                    S.extend(hat_alliance[id])
                else:
                    S.append(hat_alliance[id])
          
            values =v.cuda()
            grand = link(grand_model(x))
            null = link(null_model(x))
            # print(x.size(),values.size())
            # Evaluate explainer.
            pred, _ = evaluate_explainer(
                explainer, normalization, x, grand, null, num_players)
            
        
            # Calculate loss.
            approx = null + torch.matmul(hat_S[0], pred[0])
            loss = loss_fn(approx, values)

            # Update average.
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss, N

class DataFastSHAP:
    '''
    Wrapper around FastSHAP explanation model.

    Args:
      explainer: explainer model (torch.nn.Module).

      normalization: normalization function for explainer outputs ('none',
        'additive', 'multiplicative').
      link: link function for outputs (e.g., nn.Softmax).
    '''
    def __init__(self,
                 explainer,explainer_path, loss_path,
                 grand_model,
                 null_model, 
                 X_raw,
                 y_raw,
                 num_features,
                 normalization='additive',
                 link=None):
   
        self.explainer = explainer
        self.explainer_path = explainer_path
        self.loss_path = loss_path
        self.grand_model = grand_model #grand model
        self.null_model = null_model #null model CNN()
        self.X_raw = X_raw
        self.y_raw = y_raw
        self.num_players = num_features
        self.null = None
        if link is None or link == 'none':
            self.link = nn.Identity()
        elif isinstance(link, nn.Module):
            self.link = link
        else:
            raise ValueError('unsupported link function: {}'.format(link))

        # Set up normalization.
        if normalization is None or normalization == 'none':
            self.normalization = None
        elif normalization == 'additive':
            self.normalization = additive_efficient_normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))
        
    def train(self,
              predict_,
              train_data,
              val_data,
              val_data_label,
              class_lr,
              batch_size,
              num_samples,
              max_epochs,
              stop_epoch = 50,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=False,
              validation_samples=None,
              lookback=10,
              training_seed=None,
              validation_seed=None,
              num_workers=0,
              bar=False,
              verbose=False):
        '''
        Train explainer model.

        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          batch_size: minibatch size.
          num_samples: number of training samples.
          max_epochs: max number of training epochs.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          eff_lambda: lambda hyperparameter for efficiency penalty.
          paired_sampling: whether to use paired sampling.
          validation_samples: number of samples per validation example.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          num_workers: number of worker threads in data loader.
          bar: whether to show progress bar.
          verbose: verbosity.
        '''
        # Set up explainer model.
        explainer = self.explainer
        num_players = self.num_players
        null_model = self.null_model #null model
        grand_model = self.grand_model #grand model

        link = self.link
        normalization = self.normalization
        explainer.train()
        device = next(explainer.parameters()).device

        # Verify other arguments.
        if validation_samples is None:
            validation_samples = num_samples

        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            x_train = torch.tensor(train_data, dtype=torch.float32)
            train_set = TensorDataset(x_train)
        elif isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or '
                             'Dataset')

        # Set up validation dataset.
        if isinstance(val_data, np.ndarray):
            x_val = torch.tensor(val_data, dtype=torch.float32)
            val_set = TensorDataset(x_val)
        elif isinstance(val_data, torch.Tensor):
            x_val = torch.tensor(val_data, dtype=torch.float32)
            val_set = TensorDataset(val_data)
        elif isinstance(val_data, Dataset):
            val_set = val_data
        else:
            raise ValueError('train_data must be np.ndarray, torch.Tensor or '
                             'Dataset')
        # Set up train loader.
        # train_set = DatasetRepeat([train_set, TensorDataset(grand_train)])
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False)

        # Setup for training.
        
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(explainer.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None
        if training_seed is not None:
            torch.manual_seed(training_seed)
        
        Train_LOSS = []
        Val_LOSS = []
        
        # predict_ = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        # 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
        # 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
        # 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        # 9, 9, 9, 9])
        ##分类函数的分组
        predict_=np.array(predict_)
        label_list = []
        for predict_label in range(len(set(predict_))):
            label_list.append(np.where(predict_ == predict_label)[0].tolist())
        print('groups:',label_list)
        # print(torch.max( grand_model(self.X_raw), 1 )[1])
        
        # Generate validation data.
        sampler = ShapleySampler(len(set(predict_)))
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        print("stop_epoch:",stop_epoch)
        val_S, val_values = generate_validation_data( self.X_raw, self.y_raw, 
            val_set, label_list, null_model, stop_epoch, sampler, link, class_lr)

        # Set up val loader.
        val_set = TensorDataset(val_data, val_data_label,  val_values)
        val_loader = DataLoader(val_set, batch_size=1, 
                                pin_memory=True, num_workers=num_workers)
        
        for epoch in tqdm(range(max_epochs)):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader
            total_loss = 0
              
            #训练子分类模型
            for x,y in batch_iter:
                
                #当前联盟
                hat_alliance = label_list
                S = []
                hat_num_players =  len(set(predict_))
                hat_sampler = ShapleySampler(hat_num_players)
                hat_S = hat_sampler.sample(1, paired_sampling=paired_sampling)
                hat_S_nozero = hat_S[0].nonzero().reshape(-1)
                for id in hat_S_nozero:
                    if isinstance(hat_alliance[id],list):   
                        S.extend(hat_alliance[id])
                    else:
                        S.append(hat_alliance[id])
                # Move to device.
                x = x.cuda()
                # print(len(S)," hat_S_nozero:", hat_S_nozero,"S:",S)
                # print("Classification model training data size",torch.tensor(self.X_raw)[S_nozero].size())
                X_S = torch.tensor(self.X_raw)[S]
                y_X = torch.tensor(self.y_raw)[S]
                train_dataset = TensorDataset(X_S,y_X)
         
                # print('子联盟数据集大小:',X_S.shape)
                #训练子服务模型
                class_train_loader=DataLoader(train_dataset, batch_size=20, shuffle=True)
                net = copy.deepcopy(null_model)
                class_optimizer=torch.optim.Adam(net.parameters(),lr=class_lr)
                loss_function=nn.CrossEntropyLoss()
                values = 0
                for ep in range(stop_epoch):
                # 记录把所有数据集训练+测试一遍需要多长时间 
                    for img, label in class_train_loader:  # 对于训练集的每一个batch
                        # print(img.shape,label)  
                        img = img.cuda()
                        label = label.cuda()
                        out = net(img)  # 送进网络进行输出
                        loss1 = loss_function( out, label ) 
                        class_optimizer.zero_grad()
                        loss1.backward()
                        class_optimizer.step()
                        # print(out.shape) 
                    # print("训练第{}个epoch".format(ep))
            
                    with torch.no_grad():
                        values += link(net(x.cuda()))
                values /= stop_epoch
                # 释放模型内存
                del net
                # 清理GPU缓存
                torch.cuda.synchronize()
                # Grand and Null coalition value.
                grand = link(grand_model(x.cuda()))
                null = link(null_model(x.cuda()))
                
                # Evaluate explainer.
                pred, total = evaluate_explainer(
                    explainer, normalization, x, grand, null, num_players)
                # print(pred.shape)
                # Calculate loss.
                approx = null + torch.cat([torch.matmul(hat_S[0].cuda(), pred[i]) for i in range(pred.size()[0])]).view(pred.size()[0],10)
                #约束同类的Shapley值要接近
   
                loss = loss_fn(approx, values) 
                if eff_lambda:
                    loss = loss + eff_lambda * loss_fn(total, grand - null)
                total_loss += loss
            
            # print(pred.shape)
            
            # print(y[0],torch.max( grand_model(x[0].resize(1,1,28,28)), 1 )[1])
            # print(sum(pred[0][label_list[9],9]).item(),pred[0][label_list[9],9])
            # Take gradient step.
            loss = total_loss
            loss.backward()
            optimizer.step()
            explainer.zero_grad()
                       
            # Evaluate validation loss.
            explainer.eval()
            mean_val_loss, N =  validate(
                val_loader, val_S,  label_list, num_players, explainer, grand_model, null_model, link,
                normalization)
            val_loss = mean_val_loss.item()*N
            explainer.train()
            
            # Save loss, print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Train total loss = {:.6f}'.format(loss))
                print('Val total loss = {:.6f}'.format(val_loss))
                print('')
                Train_LOSS.append(loss.item())
                Val_LOSS.append(val_loss)
                
            scheduler.step(val_loss)
            self.loss_list.append(val_loss)

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                if verbose:
                    print('New best epoch, loss = {:.6f}'.format(best_loss))
                    torch.save(best_model, self.explainer_path)
                    print('')
            elif epoch - best_epoch == lookback or best_loss<1.3:
                if verbose:
                    print('Stopping early at epoch = {}'.format(epoch))
                break 
        
        # Copy best model.
        for param, best_param in zip(explainer.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        explainer.eval()
        
        #画loss图
        plt.figure()

        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('epochs')    # x轴标签
        plt.ylabel('loss')     # y轴标签
        X1 = [i for i in range(1, len(Train_LOSS)+1)]
        X2 = [i for i in range(1, len(Val_LOSS)+1)]
        # print(range(1, len(Train_LOSS)+1), X1,Train_LOSS)
        # 横坐标，纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(X1, Train_LOSS, linewidth=1, linestyle="solid", color='blue', label="train loss")
        plt.plot(X2, Val_LOSS, linewidth=1, linestyle="solid", color='red', label="val loss")
        plt.legend()
        plt.title('Loss curve')
        plt.savefig("{}_loss.png".format(self.loss_path))
        plt.show()


    def shap_values(self, x, label_list, grand_model, null_model):
        '''
        Generate SHAP values.

        Args:
          x: input examples.
        '''
        # Data conversion.
        x = x.cuda()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            raise ValueError('data must be np.ndarray or torch.Tensor')

        # Ensure null coalition is calculated.
        device = next(self.explainer.parameters()).device

        
        self.null  = self.link(null_model(x))
            

        # Generate explanations.
        with torch.no_grad():
            # Calculate grand coalition (for normalization).
            if self.normalization:
                grand = self.link(grand_model(x))
            else:
                grand = None

            # Evaluate explainer.
            x = x.to(device)
            # print("2---------------")
            pred, _ = evaluate_explainer(
                self.explainer, self.normalization, x, grand, self.null,
                self.num_players)
            print('pred.shape:', pred.shape)
            Shapley = np.zeros((x.shape[0],100,10))
            for i in range(len(label_list)):
                for y in range(10):
                    for num in range(x.shape[0]):
                        Shapley[num,label_list[i],y] = pred[num,i,y].cpu()/len(label_list[i])
                
                
            print(Shapley.shape,Shapley[:,label_list[0],:].shape,pred[:,0,:].shape)  
        
        return Shapley


