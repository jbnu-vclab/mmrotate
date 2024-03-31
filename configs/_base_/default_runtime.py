# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
            type='WandbLoggerHook', #WandbLoogerHook is ok
            init_kwargs=dict(entity='tuna1210', project='PNID_Angle', name='Vanilla_roi_trans'),
            interval=50,
        ),
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 16
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
