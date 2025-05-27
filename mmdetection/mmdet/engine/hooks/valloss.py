import torch
from typing import Dict, Optional, Union
from mmengine.hooks import Hook
from mmengine.runner import Runner, autocast
from mmdet.registry import HOOKS
import numpy as np


@HOOKS.register_module()
class ValLoss(Hook):
    "Save and print valid loss info"

    def __init__(self,) -> None:
        self.loss_list = []

    def before_val(self, runner) -> None:
        # build the model
        self.model = runner.model
        self.loss_list.clear()

    def after_val_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """
            Figure every loss base self.loss_list and add the output information in logs.
        """
        if len(self.loss_list) > 0:
            loss_log = {}
            total_losses = []
            for lossInfo in self.loss_list:
                total = 0.
                for k, v in lossInfo.items():
                    if 'loss' in k:
                        loss_log.setdefault(k, []).append(v)
                        
                        if isinstance(v, (list, tuple)):
                            for vv in v:
                                if isinstance(vv, torch.Tensor) and vv.numel() > 0:
                                    total += vv.reshape(())
                        elif isinstance(v, torch.Tensor) and v.numel() > 0:
                            total += v.reshape(())
                total_losses.append(total)

            for loss_name, loss_values in loss_log.items():
                flat_values = []
                for v in loss_values:
                    if isinstance(v, (list, tuple)):
                        for vv in v:
                            if isinstance(vv, torch.Tensor) and vv.numel() > 0:
                                flat_values.append(vv.reshape(())) 
                    elif isinstance(v, torch.Tensor) and v.numel() > 0:
                        flat_values.append(v.reshape(()))
                stacked = torch.stack(flat_values)
                mean_loss = torch.mean(stacked)
                runner.message_hub.update_scalar(f'val/{loss_name}_val', mean_loss)
                # runner.logger.info(f'val/{loss_name}_val: {mean_loss.item()}')
                runner.visualizer.add_scalar(f'val/{loss_name}_val', mean_loss, runner.epoch)
            val_loss = [l if type(l)==float else l.item() for l in total_losses]
            val_loss_mean = np.mean(val_loss)
            runner.message_hub.update_scalar('val/loss_val', val_loss_mean)
            runner.visualizer.add_scalar(f'val/loss_val', val_loss_mean, runner.epoch)           
        else:
            print('the model does not support validation loss!')

    # def after_val_iter(self,
    #                     runner: Runner,
    #                     batch_idx: int,
    #                     data_batch: Union[dict, tuple, list] = None,
    #                     outputs: Optional[dict] = None) -> None:
    #     """
    #     Figure the loss again
    #     Save all loss in self.loss_list.
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         with autocast(enabled=runner.val_loop.fp16):
    #             # data = self.model.data_preprocessor(data_batch, True)
    #             data = self.model.module.data_preprocessor(data_batch, True)
    #             losses = self.model._run_forward(data, mode='loss')  # type: ignore
    #             # print("Validation losses:", losses)
    #             self.loss_list.append(losses)
    #     self.model.train()

    def after_val_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: Union[dict, tuple, list] = None,
                    outputs: Optional[dict] = None) -> None:
        """Collect loss dict per val iter, with debug print."""
        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=runner.val_loop.fp16):
                data = self.model.module.data_preprocessor(data_batch, True)
                losses = self.model._run_forward(data, mode='loss')  # type: ignore

                # === Debug print start ===
                # print(f"[ValLoss Hook][Debug] Iter {batch_idx}:")
                # for k, v in losses.items():
                #     if isinstance(v, torch.Tensor):
                #         print(f"  {k}: shape={v.shape}, numel={v.numel()}, value={v.detach().cpu().numpy()}")
                #     elif isinstance(v, (list, tuple)):
                #         print(f"  {k}: list[{len(v)}]")
                #         for idx, item in enumerate(v):
                #             if isinstance(item, torch.Tensor):
                #                 print(f"    [{idx}] shape={item.shape}, numel={item.numel()}, value={item.detach().cpu().numpy()}")
                #             else:
                #                 print(f"    [{idx}] Non-tensor type: {type(item)}")
                #     else:
                #         print(f"  {k}: Unsupported type: {type(v)}")
                # === Debug print end ===

                self.loss_list.append(losses)
        self.model.train()