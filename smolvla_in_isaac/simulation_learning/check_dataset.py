import h5py

with h5py.File('../../leisaac/datasets/lift_cube.hdf5', 'r') as f:
    print('=== Dataset Structure ===')
    print(f'Эпизоды: {list(f["data"].keys())}')
    print()

    # Первый эпизод
    ep_name = list(f['data'].keys())[0]
    ep = f['data'][ep_name]
    print(f'=== Episode: {ep_name} ===')
    print(f'Keys в эпизоде: {list(ep.keys())}')
    print()

    # Actions
    if 'actions' in ep:
        print(f'Actions shape: {ep["actions"].shape}')

    # Observations
    if 'obs' in ep:
        print(f'\nObservations keys: {list(ep["obs"].keys())}')
        for key in ep['obs'].keys():
            print(f'  obs/{key} shape: {ep["obs"][key].shape}')

    # Success flag
    if 'success' in ep.attrs:
        print(f'\nSuccess: {ep.attrs["success"]}')