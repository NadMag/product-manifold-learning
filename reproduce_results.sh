# Generate all data
python generate_data.py params/params_rectangle3d.json params/params_torus.json params/params_cryo-em_x-theta_noisy.json params/params_cube.json

# Rectangle 3D
python run_expirement.py --data .\data\rectangle3d_info.pkl --configs .\configs\configs_rectangle3d.json --outdir ./rec_3d --generate_plots
