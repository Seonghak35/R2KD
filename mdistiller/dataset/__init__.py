from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100_dataloaders_trainval, get_cifar100_dataloaders_val_only, get_cifar100_dataloaders_train_only, get_cifar100_dataloaders_strong
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample, get_imagenet_dataloaders_strong
from .tiny_imagenet import get_tinyimagenet_dataloader, get_tinyimagenet_dataloader_sample

def get_dataset(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tiny_imagenet":
        if cfg.DISTILLER.TYPE in ("CRD", "CRDKD"):
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader_sample(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                    k=cfg.CRD.NCE.K,
                    )
        else:
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader(
                    batch_size=cfg.SOLVER.BATCH_SIZE,
                    val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                    num_workers=cfg.DATASET.NUM_WORKERS,
                    )
            num_classes = 200
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


def get_dataset_strong(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders_strong(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes


