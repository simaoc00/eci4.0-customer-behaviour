def get_index(data, frame_id, track_id):
    return list(data.loc[data["frame_id"] == frame_id].loc[data["track_id"] == track_id].index)[-1]

def get_item(data, frame_id, track_id, column):
    return data.loc[data["frame_id"] == frame_id].loc[data["track_id"] == track_id].loc[:, column].item()

def get_current_list(data, frame_id, track_id, column):
    return list(data.loc[data["frame_id"] <= frame_id].loc[data["track_id"] == track_id].loc[:, column].dropna())

def get_full_list(data, track_id, column):
    return list(data.loc[data["track_id"] == track_id].loc[:, column].dropna())
