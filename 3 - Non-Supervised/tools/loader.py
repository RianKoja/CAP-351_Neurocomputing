
import os

try:
    import fma_utils
except:
    import tools.fma_utils as fma_utils



def load_datasets():
    data_dir = os.path.join(os.path.dirname(__file__), '..', "dataset")
    mount_dir = os.path.join(os.path.dirname(__file__), '..', "mount")
    for dirs in [data_dir, mount_dir]:
        if not os.path.exists(dirs):
            os.mkdir(dirs)

    features = fma_utils.load(os.path.join(data_dir, "features.csv"))
    echonest = fma_utils.load(os.path.join(data_dir, "echonest.csv"))
    tracks = fma_utils.load(os.path.join(data_dir, "tracks.csv"))

    # Remove columns with attributes that are too specific, such as artist name, location and discovery rank (metadata or ranks):
    for col in echonest.columns:
        if col[1] in ['metadata', 'ranks', 'social_features']:
            echonest.drop(col, axis=1, inplace=True)

    # I'm only interested in genre for the tracks dataset:
    genres_full = tracks[('track', 'genre_top')].values.tolist()
    
    echonest_indexes = echonest.index.values.tolist()
    tracks_indexes = tracks.index.to_list()
    
    index_in_tracks = []
    genres_echo = []
    jj = 0
    for ii in range(len(tracks_indexes)):
        if tracks_indexes[ii] == echonest_indexes[jj]:
            genres_echo.append(genres_full[ii])
            jj += 1
            if jj == len(echonest_indexes):
                break

    return features, echonest, genres_full, genres_echo


if __name__ == '__main__':
    features_df, echonest_df, genres_all, genres_red = load_datasets()

    # Check if the length of echonest is the same of the reduced genres list:
    assert(len(echonest_df)==len(genres_red))

    # Check if the length of features is the same of the full genres list:
    assert(len(features_df)==len(genres_all))

    print(f'Finished {os.path.basename(__file__)}')