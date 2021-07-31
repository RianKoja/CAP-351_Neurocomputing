
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import loader, createdocument, print_table

# Use this class to simply handling the data sets without saving to variables nor reloading all the time
class datasets:
    def __init__(self):
        self.x_features = None
        self.x_echonest = None
        
    def get_x(self, mdl):
        if mdl.df_name == 'echonest':
            if self.x_echonest is None:
                self.x_echonest = mdl.scaler.transform(X=echonest.to_numpy())
            return self.x_echonest
        else:
            if self.x_features is None:
                self.x_features = mdl.scaler.transform(X=features.to_numpy())
            return self.x_features

    def get_genres(self, mdl):
        if mdl.df_name == 'echonest':
            return genres_echo
        return genres_full


# Define some auxiliary variables:
handle = datasets()
mount_dir = os.path.join(os.path.dirname(__file__), "mount")
features, echonest, genres_full, genres_echo = loader.load_datasets()

# Start building a report:
doc = createdocument.ReportDocument(title="Unsupervised Learning Exercise", user_name='Rian Koja')
doc.add_heading('Introduction', level=1)
doc.add_paragraph('In this work, several Self-Organizing Maps (SOM) are trained on two sets of data extracted from the Free Music Archive (FMA). These datasets represent numerical features extracted with different techniques from a large amount of sample music files. By varying the dimension and training parameters of the SOM network, the effects of these parameters can be roughly illustrated. Since each track has an associated primary music genre, as an application example, it is possible to check how the SOM organizes these genres in its grid. Final considerations on the effect of the training parameters and the topology of the network are given in the "Conclusion" section.')

doc.add_paragraph('To reproduce this report, the scripts "TrainSOMs.py" and "BuildReport.py" need to be rerun in this order, which might take a few hours. Prior to this, the FMA dataset needs to be redownloaded in the version avaliable on the 18th of August, 2021. The FMA files are not distributed with this code because they take up more than 1GB of disk space.')

doc.add_heading('The Data Sets', level=1)
doc.add_paragraph('The FMA is available at the following link: ')
doc.add_paragraph('https://github.com/mdeff/fma')
doc.add_paragraph('It comes with several music tracks, additional information and metadata and of particular interest two numerical datasets, "features.csv" and "echonest.csv".')

doc.add_heading('The Features Dataset', level=2)
doc.add_paragraph('The features dataset contains numerical features extracted from the audio files. The features are extracted using the following techniques:')
doc.add_paragraph('1. Spectral Centroid: The spectral centroid is a measure of the center frequency of a signal. It is calculated by taking the weighted average of the frequency magnitudes, weighted by their respective frequencies. The spectral centroid is a measure of the center frequency of a signal. It is calculated by taking the weighted average of the frequency magnitudes, weighted by their respective frequencies.')
doc.add_paragraph('2. Spectral Rolloff: The spectral rolloff is a measure of the spectral concentration. It is calculated by taking the weighted average of the frequency magnitudes, weighted by their respective frequencies. The spectral rolloff is a measure of the spectral concentration. It is calculated by taking the weighted average of the frequency magnitudes, weighted by their respective frequencies.')
doc.add_paragraph('3. Zero Crossing Rate: The zero crossing rate is a measure of the number of times a signal crosses zero. It is calculated by counting the number of times a signal crosses zero. The zero crossing rate is a measure of the number of times a signal crosses zero. It is calculated by counting the number of times a signal crosses zero.')
doc.add_paragraph('4. MFCC: The Mel Frequency Cepstrum Coefficients (MFCC) are a set of coefficients that describe the spectral content of a signal. They are calculated by taking the discrete cosine transform of the signal, and then taking the first 13 coefficients. The MFCC is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 13 coefficients.')
doc.add_paragraph('5. Chroma: The chroma is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The chroma is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('6. Tonnetz: The tonnetz is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The tonnetz is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('7. Chroma Deviation: The chroma deviation is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The chroma deviation is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('8. Tonnetz Deviation: The tonnetz deviation is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The tonnetz deviation is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('9. Chroma Variance: The chroma variance is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The chroma variance is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('10. Tonnetz Variance: The tonnetz variance is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The tonnetz variance is a measure of the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients.')
doc.add_paragraph('11. Chroma Entropy: The chroma entropy is a set of 12 frequency-weighted features that describe the spectral content of a signal. It is calculated by taking the discrete cosine transform of the signal, and then taking the first 12 coefficients. The chroma entropy is a measure of the spectral content of a signal.')

doc.add_paragraph('The library employed to extract the features is librosa. The library is available at the following link: ')
doc.add_paragraph('https://librosa.github.io/librosa/')
doc.add_paragraph('The script used to generate those features is available on the FMA GitHub repository.')

doc.add_heading('The Echonest Dataset', level=2)
doc.add_paragraph('The echonest dataset contains the metadata extracted from a subset of audio files, "Echonest" is the name of a service tat is now Spotify. The metadata includes the following features:')
doc.add_paragraph('1. Acousticness: The acousticness is a confidence measure from 0 to 1 of whether the track is acoustic. The higher the value, the more confident the algorithm is that the track is acoustic.')
doc.add_paragraph('2. Danceability: The danceability is a confidence measure from 0 to 1 of whether the track is danceable. The higher the value, the more danceable the track (eg. a song with high energy but low danceability may not actually be danceable).')
doc.add_paragraph('3. Energy: The energy is a confidence measure from 0 to 1 of whether the track is dynamically loud. The higher the value, the more dynamic the track (eg. a song with high energy but low energy).')
doc.add_paragraph('4. Instrumentalness: The instrumentalness is a confidence measure from 0 to 1 of whether the track is probably not full of actual music. The higher the value, the less likely the track is to be identified as instrumental.')
doc.add_paragraph('5. Liveness: The liveness is a confidence measure from 0 to 1 of whether the track is "live". The higher the value, the more likely the track is to be live.')
doc.add_paragraph('6. Loudness: The loudness is a confidence measure from -60 to 0 decibels (dB) of the track\'s intensity. The louder the track, the higher the number. ')
doc.add_paragraph('7. Speechiness: The speechiness is a confidence measure from 0 to 1 of whether the track is full of spoken words. The more exclusively speech-like the track (eg. talk show, audio book, poetry), the higher the value.')
doc.add_paragraph('8. Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.')
doc.add_paragraph('9. Valence: The valence is a confidence measure from 0 to 1 of the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (eg. sad, depressed, angry).')
doc.add_paragraph('10. Album Dates: The album dates are the year the album was released.')
doc.add_paragraph('11. Album Name: The album name is the name of the album the track is from.')
doc.add_paragraph('12.Artist Latitude: The artist latitude is the latitude of the artist\'s location.')
doc.add_paragraph('13. Artist Location: The artist location is the location of the artist.')
doc.add_paragraph('14. Artist Longitude: The artist longitude is the longitude of the artist\'s location.')
doc.add_paragraph('15. Artist Name: The artist name is the name of the artist.')
doc.add_paragraph('16. Release: The release is the release of the track.')
doc.add_paragraph('17. Artist Discovery Rank: The artist discovery rank is the rank of the artist in the echonest artist discovery.')
doc.add_paragraph('18. Artist Familiarity Rank: The artist familiarity rank is the rank of the artist in the echonest artist familiarity.')
doc.add_paragraph('19. Artist Hotttnesss Rank: The artist "hotttnesss" rank is a rank computed by echonest.')
doc.add_paragraph('20. Song Currency: The song currency is the currency of the song.')
doc.add_paragraph('21. Song Hotness: The song hotness is the hotness of the song.')
doc.add_paragraph('22. Song Temporal features')

doc.add_paragraph('')
doc.add_paragraph('Only the numerical and non artist-specific features are used in the analysis. The features are extracted using the tool that loads the datasets.')

doc.add_heading('The Genre Recognition Problem', level=2)
doc.add_paragraph('The genre recognition problem is to predict the genre of a song based on its audio features. Each track may have several genres, but one is assigned as the primary on the list of tracks. Using a SOM to try to predict the genre of a track is an interesting exercise, though the task itself is non-trivial and subject of ongoing research according to the paper "FMA: A DATASET FOR MUSIC ANALYSIS" by Defferrard et. al 2017. The idea for the current work is to use all data to train the SOM, then a genre is attributed to each neuron by counting which genre (when available) had the most tracks as winners in that neuron. Afterwards, a table of genres is created, and a reader may subjectively and qualitatively assess how similar these genres are expected to be.')

# Count the amount of occurrences of each genre in the lists, and print them in a table:

# Get unique elements in the list to know the genres:
genres = list(set(genres_full))
occurrences_full = [genres_full.count(aux) for aux in genres]
occurrences_echo = [genres_echo.count(aux) for aux in genres]
# Put them in a dataframe:
df_genres = pd.DataFrame({'Genre': genres, 'Samples in "Features.csv"': occurrences_full, 'Samples in "Echonest.csv"': occurrences_echo})
# Assign a RBG color to each genre: (aborted idea)
# colors = ["#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF", "#0000FF", "#FF00FF", "#00FFFF", "#FFFF00", "#FF0000", "#00FF00", "#00FFFF"]
# Pick randomly from colors so that length matches the amount of genres:
# colors = np.random.choice(colors, len(df_genres), replace=False)

# df_genres['Color'] = colors

# Create a table with this dataframe and add it to the document:
print_table.render_mpl_table(df_genres, font_size=12)
doc.add_fig()


doc.add_paragraph('Notably, many tracks don\'t have a defined genre. And some categories such as \'Experimental\' are possibly overly represented in this dataset. When cheking which genre is predominant in a neuron, the unidentified genre "nan" is ignores, giving priority to the second most occuring genre in that neuron. If only "nan" classified tracks or no tracks at all are winners on that given neuron, then "None" will be displayed as the most common genre.')

doc.add_heading('Training SOMs', level=2)
doc.add_paragraph('The parameters in training a Self Organizing-Map are the width and length of the used neuron grid, hereinafter referred as "x" and "y", the input length, which is defined by the amount of features used, the spread of the neighborhood function denoted by "sigma" and the learning rate, which modulates the magnitute of weight updates. A maximum of twenty thousand iterations are allowed.')

doc.add_paragraph('To illustrate the effects of those parameters, they will be varied when training SOMs for both datasets. In each of the following sections, the used parameters are presented, along with the topographic error of the trained network, the resulting distance map, and the table with the most commonly associated genre of each neuron. Further comments are presented afterwards.')

# Create a figure to add all distance maps in a single figure:
for i in range(9):  # Extra figure to try to prevent a bug
    fig = plt.figure(i, figsize=(10, 10))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

# Iterate over .pckl files in mount folder:
i = -1
for filename in os.listdir(mount_dir):
    i += 1
    if filename.endswith(".pckl"):
        # Read .pckl file:
        with open(os.path.join(mount_dir, filename), "rb") as file:
            mdl = pickle.load(file)

        doc.add_heading(mdl.name, level=2)
        doc.add_heading("Model Summary", level=3)
        doc.add_paragraph(f'Dataset used: "{mdl.df_name}"')
        doc.add_paragraph(f'x = {mdl.x}')
        doc.add_paragraph(f'y = {mdl.y}')
        doc.add_paragraph(f'sigma = {mdl.sigma}')
        doc.add_paragraph(f'learning_rate = {mdl.learning_rate}')
        doc.add_paragraph(f'Topographic Error = {100*mdl.topographic_error:.2f}%')

        # Get normalized data:
        X = handle.get_x(mdl)

        # Plot the heatmap:
        doc.add_heading("Distance Map", level=3)
        heat_map = mdl.som.distance_map().T

        fig, ax = plt.subplots()
        im = ax.imshow(heat_map, cmap=plt.cm.hot, interpolation='none')
        cbar = fig.colorbar(im, extend='max')
        plt.title(f'Distance Map \nSOM for "{mdl.df_name}" x={mdl.x} y={mdl.y} sigma={mdl.sigma} learning_rate={mdl.learning_rate}')
        plt.draw()
        doc.add_fig()
        plt.close()

        fig = plt.figure(i//9)
        ax = fig.add_subplot(3, 3, (i % 9)+1)
        ax.imshow(heat_map, cmap=plt.cm.hot, interpolation='none')
        plt.ylabel(f'LR = {mdl.learning_rate}')
        plt.title(f'sigma = {mdl.sigma}')

        doc.add_heading("Genre Classification", level=3)
        print_table.render_mpl_table(mdl.genres_df)
        doc.add_fig()
        plt.close()

doc.add_heading("Effect of SOM parameters", level=2)
doc.add_paragraph("For better comparison, the distance maps of the networks are presented in the figures below, which will allow comparing the effects of varying the parameters under the same meta-parameters (i.e. number of neurons).")
for jj in range(8):
    if jj <4:
        df_name = 'Echonest'
    else:
        df_name = 'Features'
    doc.add_heading(f'Distance maps for size x = y = {((jj%4)+1)*10} on {df_name} Data Set', level=3)
    fig = plt.figure(jj)
    plt.tight_layout()
    plt.draw()
    doc.add_fig()

doc.add_paragraph("The higher the value set for sigma, the less the neighboring neurons are updated when a new input vector is presented. This induces the SOM to more localized in the input space. It is particularly clear in low dimension maps that the distances are higher between the trained neurons when sigma is 5, a situation in which the whole map looks brighter. Conversely, when sigma has a lower value, more points on the map remain close, which indicates less differentiation between them.")

doc.add_paragraph("The higher the value set for learning rate, the more the SOM is able to adapt to the input vectors. It somewhat counterbalances the effect of sigma, in the sense that a map trained with a higher learning rate will look more homogenous, as if it was trained with a lower sigma, although in the range of values testes, the effect is less pronounced.")

doc.add_paragraph('The size of the network, expressed by the meta-parameters x and y seems to have an effect first on how distinguished each neuron must be, since there is little space to fit all data points. On a large grid, however the common genres get spread all over the classification table, although quite notable, the "Old-Time/Historic" genre seems concentrated on a specific block within the matrix. To better visualize large tables, one option is to copy the figure from Word into a PowerPoint presentation. For smaller maps, is is easier to visualize the tables on the Word document, and the effect of the large number of samples from certain genres such as "Rock" and "Electronic", but mostly "Experimental" can already be seen. Because these groups can be very diverse within themselves while sometimes similar to others, many less represented genres will not even appear. This is a shortcoming of the crude technique applied, the classification was done completely oblivious to these labels, while later a similarity between these genres and the mapped features is found. Nonetheless, often the "Old-Time/Historic" genre will be clustered in a block, close to "Classical" and "Folk"  which does seem to imply a level of similarity between those genres. Particularly in the case of x=y=10 sigma=3 and learning_rate=0.9 for the echonest dataset, "Folk" and "Old-Time/Historic" were placed far apart.')

doc.add_paragraph('Interestingly enough, most maps in higher dimensions left bright spots, where distances to neighbooring neurons are large, but these regions are spots, not curves, in the sense that the network is not clustering similar samples and leaving a division between them, but possibly the mesh of neurons is not stretching in those neurons, meaning that the mesh could have been initialized in a manner that certain neurons are rarely updated. This is naturally alleviated for higher values and sigma and learning rates.')

doc.add_heading("Conclusion", level=2)
doc.add_paragraph('As stated earlier, the automated classification of music is a difficult task, and a subject of ongoing research, and probably would require much more balanced datasets to have a reasonable looking classification. However, the usability of a non-supervised learning method for a very complex task could be illustrated, which meets the goals proposed for the current work.')

doc.finish()

# Launch the created document:
if os.name == 'nt':
    os.system(f'start "" "{os.path.join(mount_dir, doc.file_name)}"')
else:
    os.system(f'xdg-open "{os.path.join(mount_dir, doc.file_name)}"')

print(f'Finished {os.path.basename(__file__)}')