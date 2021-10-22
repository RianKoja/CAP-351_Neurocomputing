########################################################################################################################
# Creates a report explaining the current work and presenting obtained results.
#
# Written by Rian Koja to publish in a GitHub repository with specified license.
########################################################################################################################

# Standard imports:
import glob
import os
import pickle

# PyPI imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Local imports
from tools import createdocument, PlotData, classes, manipulations


# Define constants:
config = classes.Config()

#model1 = "Model(36,20,20,20,1).pckl"
#model1 = "Model(36,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,400,1).pckl" # Fails
#model2 = "Model(36,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,1).pckl" # Fails
#model2 = "Model(36,50,50,50,50,50,50,50,50,50,50,50,50,50,50,1).pckl" # Fails
#model2 = "Model(36,40,40,40,40,40,40,40,1).pckl" # works
#model2 = "Model(36,400,400,400,1).pckl" #works
model2 = "Model(36,600,600,600,600,600,600,600,600,1).pckl"
model1 = model2

# Start building a report:
mount_dir = os.path.join(os.path.dirname(__file__), "mount")
results_dir = os.path.join(os.path.dirname(__file__), "Results")
data_folder = os.path.join(os.path.dirname(__file__), "ProcessedData")
models_folder = os.path.join(os.path.dirname(__file__), "Models")
latex_folder = os.path.join(os.path.dirname(__file__), "LaTeX_report")

os.makedirs(mount_dir, exist_ok=True)

doc = createdocument.ReportDocument(title="Draft report to preview figures", user_name="Rian Koja")

# Load a data frame from the .csv file with the scores:
df_scores = pd.read_csv(os.path.join(results_dir, "Results_" + ".csv"))

doc.add_heading("Aggregated Results", level=2)
PlotData.plot_correlation(df_scores, "MSE Train", "MSE Future")
doc.add_fig()

#for ticker in config.target_tickers:
#    # Load data used for training, testing and validation:
#    data_file = os.path.join(data_folder, ticker + ".csv")
#    mdl_empty = classes.ModelSpecs()
#    # Get data
#    x, y, dt = mdl_empty.get_x_y(data_folder, ticker, config.reference_tickers)
#    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
#    # Free unused variables to reduce memory usage:
#    del x, y, dt

doc.add_heading("Sample Results", level=2)

doc.add_heading("First Model Results", level=3)
# Load model for ticker from saved pickle file:
with open(os.path.join(models_folder, model1), "rb") as fid:
    mdl = pickle.load(fid)
mdl.load_rnn(models_folder)
PlotData.plot_training(mdl)
doc.add_fig()
plt.close()

for ticker in config.target_tickers:
    doc.add_heading(ticker, level=4)
    x, y, dt = mdl.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
    PlotData.plot_predictions(ticker, mdl, x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end)
    doc.add_fig()
    plt.close()

del x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end

doc.add_heading("Strategy Returns on Test Data", level=4)

roics_ini = [0 for ticker in config.target_tickers]
roic_lines = ""
thresholds = [0.01, 0.03, 0.05, 0.07, 0.09]
for ticker in config.target_tickers:
    doc.add_heading(ticker, level=4)
    x, y, dt = mdl.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
    #predict_output_trn = mdl.predict(x_trn).numpy().flatten()
    predict_output_ini = mdl.predict(x_ini).numpy().flatten()
    #predict_output_end = mdl.predict(x_end).numpy().flatten()
    roics_ini_ticker = PlotData.strategy_gain_plot(x_ini, predict_output_ini, y_ini, dt_ini, doc,
                                                   f"Past (Test) Data with topology: ({mdl.topology}) and ticker {ticker}",
                                                   thresholds)
    roics_ini = [a+b/len(config.target_tickers) for a, b in zip(roics_ini, roics_ini_ticker)]
    roic_lines += ticker + " &" + ("&".join([f'{roic*100:0.2f}\\%' for roic in roics_ini_ticker])) + "\\\\ \n"

line_thresholds = "& \\textbf{\\textit{" + ("& \\textbf{\\textit{".join([f'{r*100:0.2f}\\%'+'}}' for r in thresholds]))
line_thresholds = "\\textbf{Ticker} " + line_thresholds + "\\\\ \n"

line_total = "\\textbf{Average}: &" + ("&".join([f'{roic*100:0.2f}\\%' for roic in roics_ini])) + "\\\\ \n"

text_roics = """
\\newcommand{\\makereturnsini}{
\\begin{table}[htbp]\label{tab:ini}
\\caption{Average ROICs for topology (""" + mdl.topology + """) on Test Data}
\\begin{center}
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{ROIC}&\\multicolumn{5}{|c|}{\\textbf{Required Expected Return}} \\\\
\\cline{2-6} \n""" + line_thresholds + "\\hline \n" + roic_lines + "\\hline" + line_total + """\\hline
\\end{tabular}
\\label{tab1}
\\end{center}
\\end{table}
}"""

with open(os.path.join(latex_folder, "funcs", "makereturnsini.tex"), 'w') as fid:
    print(text_roics, file=fid)

doc.add_heading("Strategy Returns on Training Data", level=4)
roics_trn = [0 for ticker in config.target_tickers]
roic_lines = ""
for ticker in config.target_tickers:
    doc.add_heading(ticker, level=4)
    x, y, dt = mdl.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
    predict_output_trn = mdl.predict(x_trn).numpy().flatten()
    #predict_output_ini = mdl.predict(x_ini).numpy().flatten()
    #predict_output_end = mdl.predict(x_end).numpy().flatten()
    roics_trn_ticker = PlotData.strategy_gain_plot(x_trn, predict_output_trn, y_trn, dt_trn, doc,
                                                   f"Training Data with topology: ({mdl.topology}) and ticker {ticker}",
                                                   thresholds)
    roics_trn = [a+b/len(config.target_tickers) for a, b in zip(roics_trn, roics_trn_ticker)]
    roic_lines += ticker + " &" + ("&".join([f'{roic*100:0.2f}\\%' for roic in roics_trn_ticker])) + "\\\\ \n"

line_thresholds = "& \\textbf{\\textit{" + ("& \\textbf{\\textit{".join([f'{r*100:0.2f}\\%'+'}}' for r in thresholds]))
line_thresholds = "\\textbf{Ticker} " + line_thresholds + "\\\\ \n"

line_total = "\\textbf{Average}: &" + ("&".join([f'{roic*100:0.2f}\\%' for roic in roics_trn])) + "\\\\ \n"

text_roics = """
\\newcommand{\\makereturnstrn}{
\\begin{table}[htbp]\label{tab:trn}
\\caption{Average ROICs for topology (""" + mdl.topology + """) on Training Data}
\\begin{center}
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
\\textbf{ROIC}&\\multicolumn{5}{|c|}{\\textbf{Required Expected Return}} \\\\
\\cline{2-6} \n""" + line_thresholds + "\\hline \n" + roic_lines + "\\hline" + line_total + """\\hline
\\end{tabular}
\\label{tab1}
\\end{center}
\\end{table}
}"""

with open(os.path.join(latex_folder, "funcs", "makereturnstrn.tex"), 'w') as fid:
    print(text_roics, file=fid)


doc.add_heading("Strategy Returns on Validation Data", level=4)
#roics_end = PlotData.strategy_gain_plot(x_end, mdl.predict_output_end, y_end, dt_end, doc, f"Future (Validation) Data and topology: {mdl.layers}")
#doc.add_paragraph(f'The strategy when trading on the most recent data yielded an annualized ROICs of {100*roics_end[0]:.2f}%, {100*roics_end[1]:.2f}%, and {100*roics_end[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order. This is partially due to the fact that for most of the concerned period the asset just fell in value, rather than increase, but these are nonetheless smaller losses than holding the asset for the whole period.')

doc.add_heading("Very Deep Network", level=3)
#doc.add_paragraph('After training the simplest models for a roughly a day, the topology (7, 8, 9, 1) emerged as a best performer at that point. This section shows the results obtained for it with some comparison with the reference result obtained for the simplest model. Notably, this network did not take significantly long to train, but the time required to reach this model is mostly related to the amount of models with poor topologies that were trained in the meantime. At some point, models with hidden layers smaller than 4 neurons were not tested.')

with open(os.path.join(models_folder, model2), "rb") as fid:
    mdl = pickle.load(fid)
mdl.load_rnn(models_folder)

PlotData.plot_training(mdl)
doc.add_fig()
plt.close()

for ticker in config.target_tickers:
    doc.add_heading(ticker, level=4)
    x, y, dt = mdl.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
    PlotData.plot_predictions(ticker, mdl, x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end)
    doc.add_fig()
    plt.close()

doc.add_heading("One week of Training Results", level=3)
# find most recent file in models folder:
latest_model = max(glob.iglob(os.path.join(os.path.dirname(__file__), "models", "*.pckl")), key=os.path.getctime)

with open(latest_model, "rb") as fid:
    mdl = pickle.load(fid)
mdl.load_rnn(models_folder)
PlotData.plot_training(mdl)
doc.add_fig()

for ticker in config.target_tickers:
    doc.add_heading(ticker, level=4)
    x, y, dt = mdl.get_x_y(data_folder, ticker, config.reference_tickers)
    x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end = manipulations.split_sets(x, y, dt)
    PlotData.plot_predictions(ticker, mdl, x_trn, y_trn, dt_trn, x_ini, y_ini, dt_ini, x_end, y_end, dt_end)
    doc.add_fig()
    plt.close()

doc.add_heading("Strategy Returns on Test Data", level=4)
#roics_ini = PlotData.strategy_gain_plot(x_ini, mdl.predict_output_ini, y_ini, dt_ini, doc, f"Past/Test Data for topology: {mdl.layers}")
#doc.add_paragraph(f'The strategy when trading on the past data yielded an annualized ROICs of {100*roics_ini[0]:.2f}%, {100*roics_ini[1]:.2f}%, and {100*roics_ini[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

doc.add_heading("Strategy Returns on Training Data", level=4)
#roics_trn = PlotData.strategy_gain_plot(x_trn, mdl.predict_output_trn, y_trn, dt_trn, doc, f"Training Data for topology: {mdl.layers}")
#doc.add_paragraph(f'The strategy when trading on the training data yielded an annualized ROICs of {100*roics_trn[0]:.2f}%, {100*roics_trn[1]:.2f}%, and {100*roics_trn[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

doc.add_heading("Strategy Returns on Validation Data", level=4)
#roics_end = PlotData.strategy_gain_plot(x_end, mdl.predict_output_end, y_end, dt_end, doc, f"Future/Validation Data for topology: {mdl.layers}")
#doc.add_paragraph(f'The strategy when trading on the most recent data yielded an annualized ROICs of {100*roics_end[0]:.2f}%, {100*roics_end[1]:.2f}%, and {100*roics_end[2]:.2f}%, and {100*roics_ini[3]:.2f}%, and {100*roics_ini[4]:.2f}%, and {100*roics_ini[5]:.2f}%, in decreasing threshold order.')

#mdl.plot_model()

#doc.add_heading("Conclusion", level=2)
#doc.add_paragraph(f"The tickers {str(config.target_tickers)} were chosen for a myriad of reasons, one of them being the fact that the lack of a consistent upward trend should prevent good results coming from checking a broken clock at the right time of the day, meaning, good results for this ticker would be unlikely to be the effect of a coincidence or the fact that the proposed strategy would always be profitable no matter the accuracy of the prediction if the asset would always appreciate over the period of twenty financial days. As explained in the introduction, the prediction problem for a financial asset is inherently difficult, and no quantitative strategy, especially one designed to be relatively simple should perform consistently over long periods of time. All that being said, the results currently obtained were significantly underwhelming, which is mostly due to the fact that as the correlation graphs show, there is little generalization capacity for the limited size of the models tested. The thresholds applied for the investment strategy do not seem to point towards there existing some sweet spot number, being too conservative when the threshold is high and though not so aggressive when the threshold is low, but nonetheless no result found for annualized return on invested capital were above a reasonable equity risk premium, let alone a proper target return rate. The exercise nonetheless allowed experimenting with time series forecasting and Long Short-Term Memory neural networks, and develop some first hand experience in implementing and training them.")


doc.finish()
# Launch the created document:
if os.name == "nt":
    os.system(f'start "" "{os.path.join(mount_dir, doc.file_name)}"')
else:
    os.system(f'xdg-open "{os.path.join(mount_dir, doc.file_name)}"')
print("Finished", os.path.basename(__file__))
