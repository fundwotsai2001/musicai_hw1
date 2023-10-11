from spleeter.separator import Separator
import os
import dill
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def create_dataset(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = os.listdir(artist_folder)
    # iterate through all artists, albums, songs and find mel spectrogram
    print(os.listdir(artist_folder))
    print(artists)
    separator = Separator('spleeter:2stems-16kHz')
    for artist in artists:
        print(artist)
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)

            for song in album_songs:
                song_path = os.path.join(album_path, song)
                splited_song = f'{song}_spleet'
                song_dir = os.path.join(album_path, splited_song)
                if os.path.isfile(song_path)==1 and os.path.isdir(song_dir)==0 :                  
                    input_audio_path = song_path
                    output_directory = os.path.join(album_path, f'{song}_spleet')
                    separator.separate_to_file(input_audio_path, output_directory)
                    print(f'{song} complete')
                # # Create mel spectrogram and convert it to the log scale
                # y, sr = librosa.load(song_path, sr=sr)
                # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                #                                    n_fft=n_fft,
                #                                    hop_length=hop_length)
                # log_S = librosa.amplitude_to_db(S, ref=1.0)
                # data = (artist, log_S, song)

                # # Save each song
                # save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                # with open(os.path.join(save_folder, save_name), 'wb') as fp:
                #     dill.dump(data, fp)
def create_testing_dataset(artist_folder='artist20_testing_data', save_folder='song_test_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = os.listdir(artist_folder)
    separator = Separator('spleeter:2stems')
    for song in artists:
        song_path = os.path.join(artist_folder, song)
        input_audio_path = song_path
        output_directory = os.path.join(artist_folder, f'{song}_spleet')
        separator.separate_to_file(input_audio_path, output_directory)
        print(f'{song} complete')
        # print(song_path)
        # Create mel spectrogram and convert it to the log scale
        # y, sr = librosa.load(song_path, sr=sr)
        # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
        #                                     n_fft=n_fft,
        #                                     hop_length=hop_length)
        # print("s",S.shape)
        # log_S = librosa.amplitude_to_db(S, ref=1.0)
        # print("log_S",log_S.shape)
        # data = (log_S)

        # # Save each song
        # save_name = song
        # with open(os.path.join(save_folder, save_name), 'wb') as fp:
        #     dill.dump(data, fp)
if __name__ == '__main__':
    # create_dataset(artist_folder='artists', save_folder='song_data',
    #                 sr=16000, n_mels=128, n_fft=2048,
    #                 hop_length=512)
    create_testing_dataset(artist_folder='artist20_testing_data', save_folder='song_test_data',
               sr=16000, n_mels=128,
               n_fft=2048, hop_length=512)