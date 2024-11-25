import os
import time
import torch
import argparse
import pandas as pd
from tqdm import tqdm

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--user_item_data', default=None, type=str, help='Path to user-item interaction data')
parser.add_argument('--rating_data', default=None, type=str, help='Path to user-item rating data')

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    u2i_index, i2u_index = build_index(args.dataset)
    
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device) 
    
    print("Loading model...")
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            print("Model loaded successfully.")
        except Exception as e:
            print(f'Failed loading state_dicts: {e}')
            print('Please check the file path and ensure it is a valid state_dict file.')
            exit(1)
            
    # Set the model to evaluation mode for inference
    model.eval()
    
    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        if args.inference_only:
            if args.user_item_data is not None and args.rating_data is not None:
                user_item_df = pd.read_csv(args.user_item_data, sep="\s+", header=None, names=['user_id', 'item_id'])
                rating_df = pd.read_csv(args.rating_data, sep="::", header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
                user_item_df = pd.merge(user_item_df, rating_df[['user_id', 'item_id', 'rating']], on=['user_id', 'item_id'], how='left')

                user_embeddings = []
                print("Processing user-item interactions...")

                for _, row in tqdm(user_item_df.iterrows(), total=len(user_item_df), desc="User-Item Interaction Processing"):
                    user_id = row['user_id']
                    item_id = row['item_id']
                    rating = row['rating']

                    # Prepare user interaction history
                    user_seq = user_train[user_id] if user_id in user_train else []
                    if len(user_seq) > args.maxlen:
                        user_seq = user_seq[-args.maxlen:]
                    elif len(user_seq) < args.maxlen:
                        user_seq = [0] * (args.maxlen - len(user_seq)) + user_seq

                    # Convert user sequence to tensor
                    user_seq = torch.LongTensor(user_seq).unsqueeze(0).to(args.device)  # Shape: (1, maxlen)

                    # Get the index of the target item in the user's history
                    if item_id in user_seq[0]:
                        target_index = (user_seq[0] == item_id).nonzero(as_tuple=True)[0].item()
                    else:
                        # If item_id is not in the sequence, skip this iteration
                        continue

                    # Generate embedding for the target item within the user sequence context
                    user_item_embeddings = model.log2feats(user_seq)  # Shape: (1, maxlen, hidden_units)
                    final_item_embedding = user_item_embeddings[:, target_index, :]  # Extract embedding for the target item

                    user_embeddings.append([user_id, item_id, rating] + final_item_embedding.squeeze(0).cpu().numpy().tolist())

                # Create DataFrame and save
                user_item_embedding_df = pd.DataFrame(user_embeddings, columns=['user_id', 'item_id', 'rating'] + [f'emb_{i}' for i in range(args.hidden_units)])
                user_item_embedding_path = os.path.join(args.dataset + '_' + args.train_dir, 'user_item_embeddings.csv')
                user_item_embedding_df.to_csv(user_item_embedding_path, index=False)
                print(f"User-Item Interaction Data with Embeddings saved to {user_item_embedding_path}")
            else:
                # Save the general item embeddings (without user history)
                print("Generating general item embeddings...")
                item_indices = torch.arange(1, itemnum + 1, device=args.device)  # Shape: (itemnum,)
                item_embeddings = model.item_emb(item_indices)  # Initial item embeddings (itemnum, hidden_units)
                item_embeddings *= model.item_emb.embedding_dim ** 0.5

                pos_indices = torch.arange(1, args.maxlen + 1, device=args.device)  # Shape: (maxlen,)
                pos_emb = model.pos_emb(pos_indices)  # Positional embeddings (maxlen, hidden_units)
                
                # Expand positional embedding to match item embedding shape
                pos_emb = pos_emb.unsqueeze(0).expand(item_embeddings.size(0), -1, -1)  # (itemnum, maxlen, hidden_units)
                item_embeddings = item_embeddings.unsqueeze(1) + pos_emb  # Add positional embeddings
                item_embeddings = model.emb_dropout(item_embeddings)  # Apply dropout

                # Pass item embeddings through transformer blocks
                for i in range(len(model.attention_layers)):
                    item_embeddings = torch.transpose(item_embeddings, 0, 1)  # (maxlen, itemnum, hidden_units)
                    Q = model.attention_layernorms[i](item_embeddings)
                    mha_outputs, _ = model.attention_layers[i](Q, item_embeddings, item_embeddings)
                    item_embeddings = Q + mha_outputs
                    item_embeddings = torch.transpose(item_embeddings, 0, 1)  # (itemnum, maxlen, hidden_units)

                    item_embeddings = model.forward_layernorms[i](item_embeddings)
                    item_embeddings = model.forward_layers[i](item_embeddings)

                # Extract final item embeddings
                final_item_embeddings = model.last_layernorm(item_embeddings)  # (itemnum, maxlen, hidden_units)
                final_item_embeddings = final_item_embeddings[:, -1, :]  # Take the last position embedding
                
                # Save the embeddings to a file
                item_ids = torch.arange(1, itemnum + 1).cpu().numpy()
                item_embeddings_np = final_item_embeddings.cpu().numpy()
                df = pd.DataFrame(item_embeddings_np)
                df.insert(0, 'item_id', item_ids)
                df = df.sort_values(by='item_id')  # Ensure item IDs are sorted
                
                embedding_path = os.path.join(args.dataset + '_' + args.train_dir, 'item_embeddings.csv')
                df.to_csv(embedding_path, index=False)
                print(f"Final Item Embeddings saved to {embedding_path}")
            
    sampler.close()
    print("Done")
