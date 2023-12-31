import os
import plot_counterfactual
import plot_reconstruction
import plot_samples
import plot_umap
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import MAKE_DATA
from erm import ERM
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task, EvalStage
from vae import VAE


def make_data(args, eval_stage):
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size, args.eval_batch_size)
    if eval_stage is None:
        data_eval = None
    elif eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif eval_stage == EvalStage.VAL:
        data_eval = data_val
    else:
        assert eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val, data_test, data_eval


def ckpt_fpath(args, task):
    return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


def make_model(args, task, eval_stage):
    is_train = eval_stage is None
    if task == Task.ERM:
        if is_train:
            return ERM(args.lr, args.weight_decay)
        else:
            return ERM.load_from_checkpoint(ckpt_fpath(args, task))
    elif task == Task.VAE:
        return VAE(task, args.parent_size, args.child_size, args.h_sizes, args.prior_reg_mult, args.init_sd,
            args.kl_anneal_epochs, args.lr, args.weight_decay)
    else:
        assert task == Task.CLASSIFY
        return VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=task)


def run_task(args, task, eval_stage):
    pl.seed_everything(args.seed)
    data_train, data_val, data_test, data_eval = make_data(args, eval_stage)
    model = make_model(args, task, eval_stage)
    if task == Task.ERM:
        if eval_stage is None:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, task.value), name='', version=args.seed),
                callbacks=[
                    ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
                max_epochs=args.n_epochs,
                deterministic=True)
            trainer.fit(model, data_train, [data_val, data_test])
        else:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value), name='', version=args.seed),
                max_epochs=1,
                deterministic=True)
            trainer.test(model, data_eval)
    elif task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value), name='', version=args.seed),
            callbacks=[
                ModelCheckpoint(monitor='val_acc', mode='max', filename='best')],
            max_epochs=args.n_epochs,
            deterministic=True)
        trainer.fit(model, data_train, [data_val, data_test])
    else:
        assert task == Task.CLASSIFY
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, task.value, eval_stage.value), name='', version=args.seed),
            max_epochs=1,
            deterministic=True,
            inference_mode=False)
        trainer.test(model, data_eval)


def main(args):
    if args.task == Task.ALL:
        run_task(args, Task.VAE, None)
        run_task(args, Task.CLASSIFY, EvalStage.VAL)
        run_task(args, Task.CLASSIFY, EvalStage.TEST)
        plot_counterfactual.main(args)
        plot_reconstruction.main(args)
        plot_samples.main(args)
        plot_umap.main(args)
    elif args.task == Task.ERM:
        run_task(args, Task.ERM, None)
        run_task(args, Task.ERM, EvalStage.VAL)
        run_task(args, Task.ERM, EvalStage.TEST)
    else:
        run_task(args, args.task, args.eval_stage)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, choices=list(MAKE_DATA.keys()))
    parser.add_argument('--task', type=Task, choices=list(Task))
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage))
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--parent_size', type=int, default=16)
    parser.add_argument('--child_size', type=int, default=16)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--prior_reg_mult', type=float, default=1e-5)
    parser.add_argument('--init_sd', type=float, default=1)
    parser.add_argument('--kl_anneal_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=100)
    main(parser.parse_args())