import logging
import os
import json
import argparse
from datetime import datetime
import time
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom imports from the src directory
from src.blocks import UNet
from src.dataset import SatelliteDataset
from src.flow_matching import FlowMatchingModel, FlowMatchingConfig
from src.DatasetBuilder import DatasetBuilder
from tqdm import tqdm


if __name__ == "__main__":
    
    # Parse tuple for dim_scales and input_shape
    def parse_tuple(s):
        try:
            # Remove brackets, spaces, convert to integers
            return tuple(map(int, s.strip().strip('()').replace(' ', '').split(',')))
        except ValueError:
            raise argparse.ArgumentTypeError("Tuple must be a string of integers separated by commas, like '1, 2, 3'.")
    
    # Set default argument
    argparser = argparse.ArgumentParser()

    # Hyper-parameters
    argparser.add_argument("--epochs", default=1000, type=int)
    argparser.add_argument("--batch-size", default=4, type=int)
    argparser.add_argument("--timesteps", default=500, type=int)
    argparser.add_argument("--sampling-timesteps", default=None, type=int)
    argparser.add_argument("--in-dim", default=5, type=int)
    argparser.add_argument("--out-dim", default=1, type=int)
    argparser.add_argument("--embed-dim", default=32, type=int)
    argparser.add_argument("--dim-scales", default=(1, 2, 4, 8), type=parse_tuple)
    argparser.add_argument("--input-shape", default=(5, 128, 128), type=parse_tuple)
    argparser.add_argument("--target-type", default="pred_x_0", type=str, choices=("pred_x_0", "pred_eps", "pred_v"))
    argparser.add_argument("--loss-type", default="l2", type=str, choices=("l1", "l2", "Hilburn_Loss"))
    argparser.add_argument("--learning-rate", default=0.0001, type=float)
    argparser.add_argument("--gf-sigmat", default=0, type=float)
    argparser.add_argument("--gf-sigma1", default=0, type=float)
    argparser.add_argument("--gf-sigma2", default=0, type=float)
    argparser.add_argument("--no-clp", action='store_true')

    # Flow Matching parameters
    argparser.add_argument("--path-type", default="optimal_transport", type=str, choices=("optimal_transport", "diffusion_like"))
    argparser.add_argument("--sigma-min", default=0.001, type=float, help="Minimum noise level for flow matching")

    # Control parameters for testing
    argparser.add_argument("--label", default="", type=str)
    argparser.add_argument("--device", default="cuda", type=str, choices=("cuda", "cpu", "mps"))
    argparser.add_argument("--num-workers", default=4, type=int)
    argparser.add_argument("--sat-files-path", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--rainfall-files-path", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--start-date", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--end-date", default="", type=str)  # Sat Dataset path
    argparser.add_argument("--max-folders", default=None, type=int)  # Sat Dataset path
    argparser.add_argument("--history-frames", default=0, type=int)  # history frames
    argparser.add_argument("--future-frame", default=0, type=int)  # predict one future frame
    argparser.add_argument("--refresh-rate", default=10, type=int)  # interval of frames
    argparser.add_argument("--model-path", default="", type=str)  # Saved model path
    argparser.add_argument("--results", default="", type=str)  # Test dataset sampling results
    argparser.add_argument("--train-model", action='store_true')  # store_true: default false=no train; store_false: default true=train
    argparser.add_argument("--retrieve-dataset", action='store_true')  # store_true: no retrieve; store_false: retrieve
    argparser.add_argument("--load-model", default="", type=str)
    args = argparser.parse_args()

    assert args.input_shape[0] == (args.in_dim - 1) * (args.history_frames + 1) + 1
    
    os.makedirs(args.model_path, exist_ok=True)
    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(name)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.model_path}/logs_{args.label}.txt")],
        )
    logger = logging.getLogger("cfmU")

    # Save Hyper-parameters
    with open(os.path.join(args.model_path, f"arguments_{args.label}.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    # log Hyper-parameters
    logger.info("Model Configuration:")
    for key, value in vars(args).items():
        logger.info(f"{key}: \"{value}\"")
    logger.info(f"Model Configuration saved to {os.path.join(args.model_path, f'arguments_{args.label}.json')}")

    # Prepare dataset
    datasetbuilder = DatasetBuilder(
        sat_path=args.sat_files_path,
        radar_path=args.rainfall_files_path,
        start_date=args.start_date,
        end_date=args.end_date,
        max_folders=args.max_folders,
        history_frames=args.history_frames,
        future_frame=args.future_frame,
        refresh_rate=args.refresh_rate,
        coverage_threshold=0.05,
        seed=96
    )
    dataset_pkl_name = "dataset_filelist.pkl"
    dataset_pkl_path = os.path.join(args.model_path, dataset_pkl_name)
    if args.retrieve_dataset:
        train_files, val_files, test_files = datasetbuilder.load_filelist(dataset_pkl_path)
        logger.info(f"Loaded existing dataset from {dataset_pkl_path}")
    else:
        train_files, val_files, test_files = datasetbuilder.build_filelist(
            save_dir=args.model_path,
            file_name=dataset_pkl_name,
            split_ratio=(0.7, 0.1, 0.2)
        )
        logger.info(f"Built new dataset to {dataset_pkl_path}")
    
    # Load dataset
    train_dataset = SatelliteDataset(files=train_files, in_dim=args.in_dim, transform=None)
    val_dataset = SatelliteDataset(files=val_files, in_dim=args.in_dim, transform=None)
    test_dataset = SatelliteDataset(files=test_files, in_dim=args.in_dim, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Log dataset info
    logger.info(f"Dataset loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Model Design
    nn_module = UNet(in_dim=(args.in_dim-1)*(args.history_frames+1)+1, out_dim=args.out_dim, embed_dim=args.embed_dim, dim_scales=args.dim_scales, no_clp=args.no_clp)
    
    model = FlowMatchingModel(
        nn_module=nn_module,
        input_shape=args.input_shape,
        config=FlowMatchingConfig(
            num_timesteps=args.timesteps,
            path_type=args.path_type,  # "optimal_transport" or "diffusion_like"
            loss_type=args.loss_type,
            sigma_min=args.sigma_min,  # minimum noise level
        ),
    )
    logger.info(f"Model {args.label} Para Nums: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Model Training
    if args.load_model:
        saved_model = os.path.join(args.model_path, args.load_model)
        checkpoint = torch.load(saved_model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['saved_epoch']
        logger.info(f"Loaded model from {saved_model}")
        logger.info(f"Optimizer state loaded from {saved_model}")
        logger.info(f"Resuming training from epoch {saved_epoch}")
    else:
        saved_epoch = 0
        logger.info("No model loaded, starting training from scratch")

    if args.train_model:
        writer = SummaryWriter(log_dir=os.path.join(args.model_path, f"logs_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M')}"))

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Initialize best validation loss
        
        logger.info(f"=== Starting Training: {args.epochs} epochs ===")
        for epoch in range(saved_epoch+1, saved_epoch+args.epochs+1):
            start_time = time.time()
            logger.info(f"=== Epoch {epoch}/{saved_epoch+args.epochs} Started ===")
            
            # Training
            model.train()
            train_loss = 0
            total_train_batches = len(train_dataloader)
            logger.info(f"Training: {total_train_batches} batches")
            
            # Use tqdm with leave=True and disable=False for GADI logging
            loop = tqdm(train_dataloader, desc=f"[Epoch {epoch}] Training", leave=True, 
                       disable=False, ncols=100, file=None)
            for batch_idx, data in enumerate(loop):
                imgs, masks, *_ = data
                imgs = imgs.permute(0, 3, 1, 2).to(args.device)
                masks = masks.permute(0, 3, 1, 2).to(args.device)
                
                optimizer.zero_grad()
                loss = model.loss(masks, cond=imgs, lead_time=args.future_frame, gf_sigmat=args.gf_sigmat).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                loop.set_postfix(loss=loss.item())
                
                # Log progress every 10% of batches
                if batch_idx % max(1, total_train_batches // 10) == 0:
                    progress = (batch_idx + 1) / total_train_batches * 100
                    logger.info(f"Training Progress: {progress:.1f}% ({batch_idx+1}/{total_train_batches}) - Loss: {loss.item():.6f}")
            
            #scheduler.step()  # Learning rate

            train_loss /= len(train_dataset)
            train_losses.append(train_loss)
            writer.add_scalar('Loss/Train', train_loss, epoch)
            logger.info(f"Training completed - Avg Loss: {train_loss:.6f}")
            
            # Validation
            model.eval()
            val_loss = 0
            val_fss = 0
            total_val_batches = len(val_dataloader)
            logger.info(f"Validation: {total_val_batches} batches")
            
            loop = tqdm(val_dataloader, desc=f"[Epoch {epoch}] Validation", leave=True,
                       disable=False, ncols=100, file=None)
            with torch.no_grad():
                for batch_idx, data in enumerate(loop):
                    imgs, masks, *_ = data
                    imgs = imgs.permute(0, 3, 1, 2).to(args.device)
                    masks = masks.permute(0, 3, 1, 2).to(args.device)
                    
                    loss = model.loss(masks, cond=imgs, lead_time=args.future_frame).mean()
                    fss = model.fss(masks, cond=imgs, lead_time=args.future_frame)
                    val_loss += loss.item() * imgs.size(0)
                    val_fss += fss.item() * imgs.size(0)
                    loop.set_postfix(val_loss=loss.item())
                    
                    # Log validation progress every 20% of batches
                    if batch_idx % max(1, total_val_batches // 5) == 0:
                        progress = (batch_idx + 1) / total_val_batches * 100
                        logger.info(f"Validation Progress: {progress:.1f}% ({batch_idx+1}/{total_val_batches}) - Val Loss: {loss.item():.6f}")

            val_loss /= len(val_dataset)
            val_losses.append(val_loss)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            val_fss /= len(val_dataset)
            writer.add_scalar('Fss/Validation', val_fss, epoch)

            elapsed = time.time() - start_time
            if epoch % 1 == 0:
                if epoch % 1 == 0:
                    logger.info(
                        f"Epoch: {epoch}/{saved_epoch+args.epochs} | "
                        f"{elapsed:.2f}s | "
                        f"Loss: {train_loss:.10f} | "
                        f"Val_loss: {val_loss:.10f} | "
                        f"Val_fss: {val_fss:.10f}"
                    )
                else:
                    logger.info(
                        f"Epoch: {epoch}/{saved_epoch+args.epochs} | "
                        f"{elapsed:.2f}s | "
                        f"Loss: {train_loss:.10f} | "
                        f"Val_loss: {val_loss:.10f}"
                    )
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'saved_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                saved_model = os.path.join(args.model_path, f"model_best_{args.label}.pt")
                torch.save(checkpoint, saved_model)
                logger.info(f"Best model saved to {saved_model} with validation loss {best_val_loss:.10f} and fss {val_fss:.10f}")
            # Save model per xx epoch
            if epoch % 10 == 0:
                checkpoint = {
                    'saved_epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }
                saved_model = os.path.join(args.model_path, f"model_epoch{epoch}_{args.label}.pt")
                torch.save(checkpoint, saved_model)
                logger.info(f"Model epoch {epoch} saved to {saved_model} with validation loss {val_loss:.10f} and fss {val_fss:.10f}")

        logger.info("=== Training Completed ===")
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Loss Curves - {args.label}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.model_path, f"Loss_Curve_{args.label}.pdf"))
        plt.close()

        checkpoint = {
            'saved_epoch': saved_epoch+args.epochs,  # saved_epoch: ran epochs
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }
        saved_model = os.path.join(args.model_path, f"model_epoch{saved_epoch+args.epochs}_{args.label}.pt")  # epoch: current
        torch.save(checkpoint, saved_model)
        logger.info(f"Final model saved to {saved_model} with validation loss {val_loss:.10f} and fss {val_fss:.10f}")
        
        writer.close()
    
    # Model Testing:
    logger.info("=== Starting Testing ===")
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(os.path.join(args.results, args.label), exist_ok=True)

    # Sample
    model.load_state_dict(checkpoint['model_state_dict'])  # Use the final model
    model.eval()

    simus_list = []
    total_test_batches = len(test_dataloader)
    logger.info(f"Testing: {total_test_batches} batches")
    
    test_loop = tqdm(test_dataloader, desc="Sampling (Test)", total=len(test_dataloader),
                    leave=True, disable=False, ncols=100, file=None)
    for idx, data in enumerate(test_loop):
        start_time = time.time()

        imgs, masks, img_times, mask_times = data
        imgs = imgs.permute(0, 3, 1, 2).to(args.device)
        samples = model.sample(cond=imgs, bsz=imgs.size(0), num_sampling_timesteps=args.sampling_timesteps, lead_time=args.future_frame, device=args.device, gf_sigma1=args.gf_sigma1, gf_sigma2=args.gf_sigma2)
        mask_gens = samples[0]  # samples_shape: (sampling_steps, bsz, out_dim, H, W)

        simus = mask_gens.permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5
        simus_list.append(simus)

        elapsed = time.time() - start_time
        test_loop.set_postfix(time=f"{elapsed:.2f}s")
        
        # Log testing progress every 10% of batches
        if idx % max(1, total_test_batches // 10) == 0:
            progress = (idx + 1) / total_test_batches * 100
            logger.info(f"Testing Progress: {progress:.1f}% ({idx+1}/{total_test_batches}) - {elapsed:.2f}s/batch")
        #if idx >= 10:
        #    logger.info("Sampling break")
        #    break

    np.save(os.path.join(args.results, args.label, f'doutputs.npy'), np.concatenate(simus_list, axis=0))
    logger.info("=== Testing Completed ===")

