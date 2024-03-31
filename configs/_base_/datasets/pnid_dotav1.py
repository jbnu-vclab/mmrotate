# dataset settings
dataset_type = 'DOTADataset'
data_root = '/data/split_ss_pnid/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

classes = ('flange', 'gate_valve', 'flanged_nozzle', 'concentric_reducer', 'utility_conenction', 'off_page_connector', 'blind_flange', 'gate_valve_with_plug', 'ball_valve', 'open_figure_8_blind', 'check_valve', 'globe_valve', 'gate_valve_with_flange', 'pipe_nozzle', 'connector', 'off_plot_connector', 'closed_figure_8_blind', 'open_drain_connectors', 'hose_connection', 'spacer', 'ball_valve_with_flange', 'butterfly_valve', 'closed_drain_connectors', 'eccentric_reducer', 'ball_valve_with_plug', 'interface_connecting_point', 'removable_spool', 't_type_strainer', 'cap', 'sample_connection', 'y_type_strainer', 'screwed_cap', 'steam_trap', 'swing_elbow', 'off_page_connector_two_way', 'pulsation_dampener', 'desuperheater', '3_way_valve', 'tie_in_symbol', 'plug', 'atmospheric_vent_with_bird_screen', 'vent_cover', 'plug_valve', 'expansion_joint', 'rotating_disc_butterfly_valve', 'per_strainer', 'flexible_hose', 'extend_steam_valve', 'dual_plate_wafer_check_valve', 'globe_valve_with_plug', 'vent_silencer', 'basket_strainer', 'blank', 'off_plot_connector_two_way', 'flame_arrestor', 'injection_quill', 'ejector_eductor', 'damper', 'angle_blowdown_valve', 'needle_valve', 'cone_strainer', 'manway', 'removable_spool_left', 'filter', 'removable_spool_right', 'double_gate_valve_with_plug', 'break_away_coupling', 'globe_valve_with_flange', 'end_gap_line', 'removable_spool_elbow', 'vendor_strainer', 'knife_valve', 'globe_valve_handwheel', 'diverter_valve', 'discrete_instruments_field_mounted', 'shared_display_and_control_primary_location_normally_accessible_to_an_operator', 'end_gap', 'plc_primary_location_normally_inaccessible_to_an_operator', 'chemical_seal_diaphragm', 'diaphragm', 'restriction_orifice', 'pressure_relief_or_safety_valve', 'plc_primary_location_normally_accessible_to_an_operator', 'discrete_hardware_interlock_field_mounted', 'cylinder', 'plc_field_mounted', 'vortex', 'handwheel', 'variable_area_flowmeter', 'shared_display_and_control_primary_location_normally_inaccessible_to_an_operator', 'discrete_instruments_auxiliary_location_normally_accessible_to_an_operator', 'pressure_reducing_regulator', 'discrete_instruments_primary_location_normally_accessible_to_an_operator', 'motor_operated', 'instrument_box', 'pilot_light_or_gauge_glass_illuminator', 'general_sight_glass', 'manual_operator', 'function_multiply', 'back_pressure_regulator', 'function_subtract', 'function_low_signal_selector', 'function_unspecified', 'function_high_signal_selector', 'venturi', 'function_derivative', 'coriolis_flowmeter', 'function_average', 'function_summation', 'restriction_orifice_multistage', 'safety_head_for_pressure_relief', 'thermal_mass_flowmeter', 'integral_orifice', 'function_special', 'orifice_valve', 'function_time', 'slide_valve', 'function_root', 'electric_motor_general', 'pump_centrifugal_type', 'vortex_breaker', 'tank_vessel', 'shell_type_e', 'vertical_drum', 'air_cooler', 'front_end_type_a', 'horizontal_drum', 'pump_positive_diaplacement', 'box_equipment_double_chain', 'real_end_type_s', 'gas_filter_bag_candle_or_cartridge_type', 'tank_vessel_with_dished_ends', 'boot_w_head', 'compressor_vacuum_pump_reciprocating_piston_type', 'in_line_static_mixer', 'blowdown_pot', 'heat_exchanger_of_double_pipe_type', 'conveyor_belt_type', 'shell_type_k', 'vent_silencer_equipment', 'sump_drum', 'shell_type_h', 'ultrasonic_type_meter', 'heat_exchanger_vendor', 'steam_turbine', 'boiler_heater', 'compressor_reciprocating_type_5', 'compressor_centrifugal_type', 'pump_vertical_wet_pit', 'shell_type_x', 'pump_gear_type', 'filter_oil_vendor', 'front_end_type_b', 'shell_type_j', 'liquid_filter_general', 'real_end_type_u', 'compressor_reciprocating_type_4a', 'vessel_with_dished_roof_and_conical_bottom', 'debutanizer', 'front_end_type_d', 'sump_vendor', 'deethanizer', 'chloride_treater', 'turbine_general', 'compressor_reciprocating_type_4b', 'compressor_reciprocating_type_2a', 'tank_flat_roof', 'vendor_heater', 'farctionator_bottom', 'oil_cooler', 'reactor_type1', 'reactor_type2', 'reactor_type3', 'heat_exchanger_packinox', 'gas_cylinder', 'annotation', 'spec_break', 'text')

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))
