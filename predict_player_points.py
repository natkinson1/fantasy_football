import numpy as np
import pandas as pd
import glob
import pystan
import re
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from player_model import PlayerModel

def get_empirical_bayes_estimates(df_emp):
    """
    Get starting values for the model based on averaging goals/assists/neither
    over all players in that position
    """
    # still not sure about this...
    df = df_emp.copy()
    #df = df[df["match_id"] != 0]
    goals = df['goals_scored'].sum()
    assists = df["assists"].sum()
    clean_sheets = df['clean_sheets'].sum()
    neither = df["neither"].sum()
    minutes = df["minutes"].sum()
    team = df["team_goals"].sum()
    total_minutes = 90 * len(df)
    neff = df.groupby("player_id").count()["goals_scored"].mean()
    a0 = neff * (goals / team) * (total_minutes / minutes)
    a1 = neff * (assists / team) * (total_minutes / minutes)
    a2 = neff * (clean_sheets / team) * (total_minutes / minutes)
    a3 = (
        neff
        * ((neither / team) - (total_minutes - minutes) / total_minutes)
        * (total_minutes / minutes)
    )
    alpha = np.array([a0, a1, a3])
    return alpha

def process_player_data(df):
    """
    transform the player dataframe, basically giving a list (for each player)
    of lists of minutes (for each match, and a list (for each player) of
    lists of ["goals","assists","neither"] (for each match)
    """
    df["neither"] = df["team_goals"] - df["goals_scored"] - df["assists"]
#     df.loc[(df["neither"] < 0), ["neither", "team_goals", "goals_scored", "assists"]] = [
#         0.0,
#         0.0,
#         0.0,
#         0.0,
#     ]
    alpha = get_empirical_bayes_estimates(df)
    print(alpha)
    
    y = df.sort_values("player_id")[["goals_scored", "assists", "neither"]].values.reshape((df["player_id"].nunique(), 
                                                                                            df.groupby("player_id").count().iloc[0][0],
                                                                                            3))

    minutes = df.sort_values("player_id")["minutes"].values.reshape((df["player_id"].nunique(),-1))

    nplayer = df["player_id"].nunique()
    nmatch = df.groupby("player_id").count().iloc[0][0]
    player_ids = np.sort(df["player_id"].unique())
    return dict(
        player_id=player_ids,
        nplayer=nplayer,
        nmatch=nmatch,
        minutes=minutes.astype("int64"),
        y=y.astype("int64"),
        alpha=alpha,
    )

def fit_player_data(data, prefix='FWD'):
    """
    fit the data for a particular position (FWD, MID, DEF)
    """
    model = PlayerModel()
    data = process_player_data(data[data['element_type'] == prefix])
    print("Fitting player model for", prefix, "...")
    fitted_model = model.fit(data)
    df = pd.DataFrame(fitted_model.get_probs())

    df["pos"] = prefix
    df = (
        df.rename(columns={"index": "player_id"})
        .sort_values("player_id")
        .set_index("player_id")
    )
    return df.reset_index()