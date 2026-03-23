import logging
import os
import torch.distributed as dist

# def create_logger(logging_dir):
#     """
#     Create a logger that writes to a log file and stdout.
#     """
#     if dist.get_rank() == 0:  # real logger
#         logging.basicConfig(
#             level=logging.INFO,
#             format='[\033[34m%(asctime)s\033[0m] %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S',
#             handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
#         )
#         logger = logging.getLogger(__name__)
#     else:  # dummy logger (does nothing)
#         logger = logging.getLogger(__name__)
#         logger.addHandler(logging.NullHandler())
#     return logger
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    Only rank 0 writes to log.txt. All ranks write to console.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger('editAR')  # 用统一 logger 名字
    # 防止重复添加 handler
    if not getattr(logger, '_is_configured', False):
        logger.setLevel(logging.INFO)
        # 只给 rank 0 加文档 handler
        if rank == 0 and logging_dir:
            os.makedirs(logging_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(logging_dir, "log.txt"), encoding='utf-8', mode='a')
            fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(fh)
        # 所有进程都加 stdout handler（彩色只给 console，不给文件）
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[\033[34m%(asctime)s\033[0m] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(ch)
        logger._is_configured = True
    return logger