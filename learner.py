import argparse
from typing import Dict
import logging
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import TeLM, KGEnPred
from regularizers import N3,  La3
import os


parser = argparse.ArgumentParser(
    description="TeLM"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name",
    default='fujian10w'
)

parser.add_argument(
    '--model', default='KGEnPred', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=3, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-2, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0.0075, type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0.01, type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--time_granularity', default=1, type=int, 
    help="Time granularity for time embeddings"
)
parser.add_argument(
    '--logs_path', default="./logs", type=str, 
    help="logs path"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            if 'lhs' in mrrs and 'rhs' in mrrs:
                m = (mrrs['lhs'] + mrrs['rhs']) / 2.
                h = (hits['lhs'] + hits['rhs']) / 2.
                res = {'MRR': {'avg': m, 'lhs': mrrs['lhs'], 'rhs': mrrs['rhs']}, 'hits@[1,3,10]': {'avg': h,'lhs': hits['lhs'], 'rhs': hits['rhs']}}
            elif 'lhs' in mrrs:
                m = mrrs['lhs']
                h = hits['lhs']
                res = {'MRR': {'avg': m, 'lhs': m}, 'hits@[1,3,10]': {'avg': h, 'lhs': h}}
            elif 'rhs' in mrrs:
                m = mrrs['rhs']
                h = hits['rhs']
                res = {'MRR': {'avg': m, 'rhs': m}, 'hits@[1,3,10]': {'avg': h, 'rhs': h}}

            return res

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          time_granularity=args.time_granularity,
         ):


    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset

    ##restore model parameters and results
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/time_granularity{:02d}/emb_reg{:.5f}/time_reg{:.5f}/'.format(rank,learning_rate,batch_size, time_granularity, emb_reg, time_reg))
    
    dataset = TemporalDataset(dataset)
    
    sizes = dataset.get_shape()
    model = {
        'TeLM': TeLM(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity),
        'KGEnPred': KGEnPred(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity)
    }[model]
    model = model.cuda()


    opt = optim.Adagrad(model.parameters(), lr=learning_rate)

    
    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg, "time_granularity =", time_granularity)

    emb_reg = N3(emb_reg)
    time_reg = La3(time_reg)
  
    # Results related
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    #os.makedirs(PATH)
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
    writer =  SummaryWriter(args.logs_path)
    global_steps = 0
    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train()
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=batch_size
        )
        
        optimizer.epoch(examples, writer, global_steps)
        
        # for name, parms in optimizer.model.named_parameters():
        #     writer.add_histogram(name, parms.data.flatten(), epoch)
        
        # writer.add_histogram("enity_emb", optimizer.model.enity_emb_tmp[:1000], epoch)
        
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
 
            valid, test, train = [
                # avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, use_left_queries=args.use_left))
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, missing_eval = "rhs"))
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            print("train: ", train['MRR'])

            # Save results

            f = open(os.path.join(PATH, 'result.txt'), 'w+')
            f.write("\n VALID: ")
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']['avg']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 10:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            if not dataset.interval:
                curve['train'].append(train)
    
                print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
            print("\t TEST : ", test)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'w+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':
    #nohup python -u learner.py --dataset fujianV2 > fujianv2_KGEnPred.log 2>&1 &
    learn()