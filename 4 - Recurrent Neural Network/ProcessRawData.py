# Process the data on RawData folder, reading each xlsx file, and creating a dataframe for each asset, combining data from different files, and saving to individual csv files in the ProcessedData folder.

# Standard imports:
import os

# PyPI imports:
import pandas as pd


print("Starting", __file__)

raw_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RawData")
processed_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ProcessedData")

# Create folder if it does not exist:
if not os.path.exists(processed_data_folder):
    os.makedirs(processed_data_folder)

tickers_file = os.path.join(raw_data_folder, "Tickers.xlsx")

# Read data from all xlsx files in RawData folder
for file in os.listdir(raw_data_folder):
    print("file = ", file)
    if file.endswith(".xlsx"):
        # Create a dictionary of dataframes, where the key is the asset name:
        tickers_dict = {}
        file_path = os.path.join(raw_data_folder, file)
        for sheet in ["Open", "Close", "High", "Low", "Volume"]:
            # print('sheet = ', sheet)
            temp_data = pd.read_excel(file_path, sheet_name=sheet)
            # Eliminate the second row, which is an artifact:
            temp_data = temp_data.iloc[1:]
            ticker_list = temp_data.columns.values.tolist()
            ticker_list.remove("Ticker:")
            # If tickers_dict is empty, populate it based on current tickers:
            if len(tickers_dict) == 0:
                for ticker in ticker_list:
                    # Create a new dataframe for each ticker, with the date and price for current sheet:
                    temp_df = temp_data[["Ticker:", ticker]]
                    # Rename the column to 'Date' and sheet name:
                    temp_df.columns = ["Date", sheet]
                    # print('d')
                    tickers_dict[ticker] = temp_df
                    # print('e')

            else:  # Add column to existing dataframes in dictionary:
                for ticker in ticker_list:
                    # print('a')
                    temp_df = temp_data[[ticker]]
                    # print('b')
                    temp_df.columns = [sheet]
                    # Add the new column to the existing dataframe:
                    tickers_dict[ticker] = tickers_dict[ticker].join(temp_df)

    # Now for each ticker, if there is no corresponding file, create is at a csv:
    for ticker in tickers_dict:
        # First eliminate lines that only contain NaN:
        tickers_dict[ticker] = tickers_dict[ticker].dropna(how="any")
        if not os.path.isfile(os.path.join(processed_data_folder, ticker + ".csv")):
            temp_df = tickers_dict[ticker]
            temp_df.drop_duplicates(inplace=True)
            temp_df.sort_values(by="Date", ascending=True, inplace=True)
            temp_df.to_csv(os.path.join(processed_data_folder, ticker + ".csv"), index=False)
        else:
            # If there is already a file, append the new dataframe to the existing file:
            existing_df = pd.read_csv(os.path.join(processed_data_folder, ticker + ".csv"))
            # convert to datetime
            existing_df["Date"] = pd.to_datetime(existing_df["Date"])
            existing_df = existing_df.append(tickers_dict[ticker])
            # Remove duplicate lines first:
            existing_df.drop_duplicates(inplace=True)
            existing_df.sort_values(by="Date", ascending=True, inplace=True)
            existing_df.to_csv(os.path.join(processed_data_folder, ticker + ".csv"), index=False)


print("Finished", __file__)
