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
    gap = (grand - null) - torch.sum(pred, dim=1)
    # print(gap.size())
    # gap = gap.detach()
    # print(gap.unsqueeze(1).size(),pred.shape[1])
    return pred + gap.unsqueeze(1) / pred.shape[1]


def multiplicative_efficient_normalization(pred, grand, null):
    '''
    Apply multiplicative efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    '''
    ratio = (grand - null) / torch.sum(pred, dim=1)
    # ratio = ratio.detach()
    return pred * ratio.unsqueeze(1)


def evaluate_explainer(explainer, normalization, x, grand, null, num_players,
                       inference=False):
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
        # Image.
        image_shape = pred.shape
     
        pred = pred.reshape(len(x),  num_players, -1)
        # pred = pred.permute(0, 2, 1)
    else:
        # Tabular.
        image_shape = None
        pred = pred.reshape(len(x), num_players, -1)
        # print(pred.shape)
    # For pre-normalization efficiency gap.
    total = pred.sum(dim=1)

    # Apply normalization.
    if normalization:
        # print(pred.shape)
        pred = normalization(pred, grand, null)
        # print(pred.shape)
    # Reshape for inference.
    if inference:
        if image_shape is not None:
            pred = pred.permute(0, 2, 1)
            pred = pred.reshape(image_shape)

        return pred

    return pred, total


def generate_validation_data(X_raw, y_raw,service_val_loader, val_set, null_model, class_epoch, sampler, num_players,
                             batch_size, link, device, class_lr):
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
        S = sampler.sample(1, paired_sampling=True)
        val_S.append(S[0])
        # print("valid one batch:",x[0].size(),S[0].size())
        # Move to device.
        # print(x.size())
        x = x[0].cuda()
        S = S[0].to(device)

        # Evaluate value function.
        S_nozero = S.nonzero().reshape(-1)

        # print("Classification model_val training data size",torch.tensor(X_raw)[S_nozero].size())
        X_S = torch.tensor(X_raw)[S_nozero]
        y_X = torch.tensor(y_raw)[S_nozero]

        train_dataset = TensorDataset(X_S,y_X)
        class_train_loader=DataLoader(train_dataset, batch_size=20, shuffle=True)
        
        net = copy.deepcopy(null_model)
        class_optimizer=torch.optim.Adam(net.parameters(),lr=class_lr)
        loss_function=nn.CrossEntropyLoss()
        accuracy_best = 0
        for ep in range(class_epoch):
        # 记录把所有数据集训练+测试一遍需要多长时间
            for img, label in class_train_loader:  # 对于训练集的每一个batch
                # print(img,label)  
                img = img.cuda()
                label = label.cuda()
                out = net( img )  # 送进网络进行输出
                loss = loss_function( out, label ) 
                class_optimizer.zero_grad()
                loss.backward()
                class_optimizer.step()
            # print("训练第{}个epoch".format(ep))
            
            # num_correct = 0  # 正确分类的个数，在测试集中测试准确率
            # for img, label in service_val_loader:       
            #     img = img.cuda()
            #     label = label.cuda()
            #     out = net( img )  # 获得输出
            #     _, prediction = torch.max( out, 1 )
            #     num_correct += (prediction == label).sum()  # 找出预测和真实值相同的数量，也就是以预测正确的数量

            # accuracy_new = num_correct.cpu().numpy() / 100 
            
            # if  accuracy_new>accuracy_best:
            #     best_ep = ep
            #     accuracy_best = accuracy_new
            # if ep-best_ep==10:
            #     # print(ep)
            #     break

        with torch.no_grad():
         
            values = link(net(x.view(1,1,28,28)))
        val_values.append(values.cpu())
        # 释放模型内存
        del net
        # 清理GPU缓存
        torch.cuda.synchronize()
    val_S = torch.stack(val_S, dim=0)
    val_values = torch.stack(val_values, dim=0)
    
    return val_S, val_values


def validate(val_loader, num_players, explainer, grand_model, null_model, link, normalization):
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
        for x,y,S,v in val_loader:
            # print("2--------------")
            # Move to device.
            x = x.cuda()
            y =y.cuda()
            S = S[0].cuda()
            values =v.cuda()
            grand = link(grand_model(x))
            null = link(null_model(x))
            # print(x.size(),y.size(),S.size(),values.size())
            # Evaluate explainer.
            pred, _ = evaluate_explainer(
                explainer, normalization, x, grand, null, num_players)
            
            # print(pred.size(),S.size())
            # Calculate loss.
            approx = null + torch.matmul(S, pred[0])
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
                 service_val_loader,
                 num_features,
                 normalization='additive',
                 link=None):
   
        self.explainer = explainer
        self.explainer_path = explainer_path
        self.loss_path = loss_path
        self.grand_model = grand_model.cuda() #grand model
        self.null_model = null_model.cuda() #null model CNN()
        self.X_raw = X_raw.cuda()
        self.y_raw = y_raw.cuda()
        self.service_val_loader =  service_val_loader
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
           
        elif normalization == 'multiplicative':
            self.normalization = multiplicative_efficient_normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))
        
    def train(self,
              train_data,
              val_data,
              val_data_label,
              batch_size,
              num_samples,
              max_epochs,
              class_lr = 0.0001,
              class_epoch = 50,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=False,
              validation_samples=None,
              lookback=20,
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
            train_set, batch_size=batch_size, shuffle=True)

        # Setup for training.
        sampler = ShapleySampler(num_players)
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
        
        # Generate validation data.
        sampler = ShapleySampler(num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        print("class_epoch:",class_epoch)
        val_S, val_values = generate_validation_data( self.X_raw, self.y_raw,self.service_val_loader, 
            val_set, null_model, class_epoch, sampler, num_players,
            batch_size * num_samples, link, device, class_lr)

        # Set up val loader.
        # print(val_data.size(), val_S.size(), val_values.size())
        val_set = TensorDataset(val_data, val_data_label, val_S, val_values)
        val_loader = DataLoader(val_set, batch_size=1, 
                                pin_memory=True, num_workers=num_workers)
        Train_LOSS = []
        Val_LOSS = []
        for epoch in tqdm(range(max_epochs)):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader
            total_loss = 0
              
            #训练子分类模型
            for x,y in batch_iter:
                # Sample S.
                S = sampler.sample(1, paired_sampling=paired_sampling)
              
                # print("Training one batch:",x.size(),S[0].size())
                # Move to device.
                x = x.cuda()
                y = y.cuda()
                S = S[0].cuda()
            
                # Evaluate value function.
                S_nozero = S.nonzero().reshape(-1)
       
                # print("Classification model training data size",torch.tensor(self.X_raw)[S_nozero].size())
                X_S = torch.tensor(self.X_raw)[S_nozero]
                y_X = torch.tensor(self.y_raw)[S_nozero]
                train_dataset = TensorDataset(X_S,y_X)
                # print(X_S.size(),y_X.size())
                class_train_loader=DataLoader(
                    train_dataset, batch_size=20, shuffle=True, 
                   )

                net = copy.deepcopy(null_model)
               
                class_optimizer=torch.optim.Adam(net.parameters(),lr=class_lr)
                loss_function=nn.CrossEntropyLoss()
                accuracy_best=0
                for ep in range(class_epoch):
                # 记录把所有数据集训练+测试一遍需要多长时间 
                    for img, label in class_train_loader:  # 对于训练集的每一个batch
                        # print(img,label)  
                        img = img.cuda()
                        label = label.cuda()
                        out = net( img )  # 送进网络进行输出
                        loss = loss_function( out, label ) 
                        class_optimizer.zero_grad()
                        loss.backward()
                        class_optimizer.step()
                    # print("训练第{}个epoch".format(ep))
                    # num_correct = 0  # 正确分类的个数，在测试集中测试准确率
                    # for img, label in self.service_val_loader:       
                    #     img = img.cuda()
                    #     label = label.cuda()
                    #     out = net( img )  # 获得输出
                    #     _, prediction = torch.max( out, 1 )
                    #     num_correct += (prediction == label).sum()  # 找出预测和真实值相同的数量，也就是以预测正确的数量

                    # accuracy_new = num_correct.cpu().numpy() / 100 
                    
                    # if  accuracy_new>accuracy_best:
                    #     best_ep = ep
                    #     accuracy_best = accuracy_new
                    #     # print(ep)
                    # if ep-best_ep==10:
                    #     break
                    
                with torch.no_grad():
                    values = link(net(x))
                
                # 释放模型内存
                del net
                # 清理GPU缓存
                torch.cuda.synchronize()
                # Grand and Null coalition value.
                grand = link(grand_model(x))
                null = link(null_model(x))
                
            
                # Evaluate explainer.
                pred, total = evaluate_explainer(
                    explainer, normalization, x, grand, null, num_players)

                # Calculate loss.
                # print(batch_size, pred.size(),null.size())
                # print(null.size(),torch.cat([torch.matmul(S, pred[i]) for i in range(batch_size)]).size())
                approx = null + torch.cat([torch.matmul(S, pred[i]) for i in range(pred.size()[0])]).view(pred.size()[0],10)
                # print(values[1])
                loss = loss_fn(approx, values)
                if eff_lambda:
                    loss = loss + eff_lambda * loss_fn(total, grand - null)
                total_loss += loss
            # Take gradient step.
            # print(link(net(self.X_raw[0].view(1,1,28,28))),torch.max(net(self.X_raw[0].view(1,1,28,28)),1)[1],self.y_raw[0].item())
            loss = total_loss
            loss.backward()
            optimizer.step()
            explainer.zero_grad()
                       
            # Evaluate validation loss.
            explainer.eval()
            mean_val_loss, N =  validate(
                val_loader,  num_players, explainer, grand_model, null_model, link,
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
                    print('New best epoch, loss = {:.6f}'.format(val_loss))
                    torch.save(best_model, self.explainer_path)
                    print('')
            elif epoch - best_epoch == lookback:
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


    def shap_values(self, x, grand_model, null_model):
        '''
        Generate SHAP values.

        Args:
          x: input examples.
        '''
        # Data conversion.
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
            pred = evaluate_explainer(
                self.explainer, self.normalization, x, grand, self.null,
                self.num_players, inference=True)

        return pred.cpu().data.numpy()


