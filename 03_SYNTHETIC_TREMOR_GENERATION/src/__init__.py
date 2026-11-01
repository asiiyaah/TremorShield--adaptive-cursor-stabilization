"""TREMORSHIELD pipeline package.

Modules:
    config        - constants and TremorConfig
    data_loader   - load_data()
    anonymize     - rename_participants()
    splitting     - split_participants(), split_training_trials(), assign_test_trials()
    tremor_model  - generate_tremor(), _unit_normals()  (Kulkarni et al., 2024 model)
    injection     - inject_tremor_into_trial()
    features      - recalculate_features()
    pipeline      - build_condition()
    validation    - validate_dataset()
    io_writer     - save_outputs()
"""
