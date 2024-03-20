# %%
import pandas as pd
import numpy as np
import time

start_time = time.time()
# %%
df_women = pd.read_excel(
    "OlympicRankings2.xlsx", sheet_name="March Women's Olympic Standings"
)


# %%
df_women.fillna(0, inplace=True)

tournaments_left = [
    "Challenge Recife, Brazil",
    "Challenge Saquarema, Brazil",
    "Challenge Guadalajara, Mexico",
    "Elite16 Tepic, Mexico",
    "Challenge Xiamen, China",
    "Elite16 Natal, Brazil",
    "Elite16 Espinho, Portugal",
    "Challenge Stare Jablonki, Poland",
    "Elite 16 Ostrava, Czech Republic",
]

df_women[tournaments_left] = pd.DataFrame(
    [[float("nan")] * len(tournaments_left)], index=df_women.index
)

df_women = df_women[~df_women["Team"].str.contains("Kelly Cheng")]
df_women.reset_index(inplace=True, drop=True)


# %%
tournament_cols = df_women.drop(["Team", "Country", "Total"], axis=1).columns


point_elite = [
    1200,
    1100,
    1000,
    900,
    760,
    760,
    760,
    760,
    600,
    600,
    600,
    600,
    460,
    460,
    460,
    460,
    400,
    400,
    400,
    340,
    340,
    340,
    340,
    340,
    340,
    340,
    340,
]


point_chal = [
    800,
    760,
    720,
    680,
    600,
    600,
    600,
    600,
    460,
    460,
    460,
    460,
    460,
    460,
    460,
    460,
    360,
    360,
    300,
    300,
    300,
    300,
    300,
    300,
    220,
    220,
    220,
    220,
    220,
    220,
    220,
    140,
    140,
    140,
    140,
    140,
    140,
]


point_elite = (point_elite) + list(np.zeros(len(df_women) - len(point_elite)))

point_chal = point_chal + list(np.zeros(len(df_women) - len(point_chal)))


point_elite = np.array(point_elite)

point_chal = np.array(point_chal)

# %%
time_dict = {
    "fill_remaining_tournaments": [],
    "calc_total_points": [],
    "check_qualified": [],
}


def fill_remaining_tournaments(df, verbose=False):
    start_time = time.time()
    for t in tournaments_left:
        if "Challenge" in t:
            random_list_outcomes_c = np.random.choice(
                point_chal, size=len(df_women), replace=False
            )

            df[t] = random_list_outcomes_c
        else:
            random_list_outcomes_e16 = np.random.choice(
                point_elite, size=len(df_women), replace=False
            )

            df[t] = random_list_outcomes_e16

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        time_dict["fill_remaining_tournaments"].append(elapsed_time)
    return df


def calc_total_points(df, verbose=False):
    start_time = time.time()
    df["new_total_points"] = None

    totals = -1 * (
        np.sum(
            np.partition(-df_women[tournament_cols].values, 12, axis=1)[:, :12], axis=1
        )
    )
    df["new_total_points"] = totals
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        time_dict["calc_total_points"].append(elapsed_time)
    return df


def create_qualified_dict(df, verbose=False):
    start_time = time.time()

    team_map = {team: [] for team in df["Team"]}
    end_time = time.time()
    elapsed_time = end_time - start_time
    return team_map


def check_qualified(df, team_map: dict, verbose=False) -> dict:
    start_time = time.time()
    df_test = df.copy()
    # df_sorted = df_test.sort_values(by="new_total_points", ascending=False)
    # df_top17 = df_sorted[:17]

    column_to_sort = df_test["new_total_points"].values
    sorted_is = np.argsort(column_to_sort)
    sorted_index = sorted_is[0:][::-1]
    index_top_17 = sorted_index[:16]
    df_top17 = df_test.iloc[index_top_17]

    while any(df_top17["Country"].value_counts() >= 3):
        country_counts = df_top17["Country"].value_counts()

        countries_over_3 = list(country_counts[country_counts > 2].index)
        if countries_over_3 != 0:
            for country in countries_over_3:
                df_top17.reset_index(drop=True, inplace=True)
                drop_index = df_top17[df_top17["Country"] == country].index[2:]

                # df_sorted.drop(drop_index, inplace=True)
                # df_top17 = df_sorted[:17]
                # print(drop_index)
                sorted_index = np.delete(sorted_index, drop_index)
                index_top_17 = sorted_index[:16]
                df_top17 = df_test.iloc[index_top_17]

    top17_teams_set = set(df_top17["Team"])
    for team in df["Team"]:
        team_map[team].append(team in top17_teams_set)

    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        time_dict["check_qualified"].append(elapsed_time)

    return team_map


# %%
def simulate(df, num_simulations=100, verbose=False):
    np.random.seed(42)
    team_qualified_map = create_qualified_dict(df, verbose)

    for _ in range(num_simulations):
        df = fill_remaining_tournaments(df, verbose)
        df = calc_total_points(df, verbose)

        team_qualified_map = check_qualified(df, team_qualified_map, verbose)

    return team_qualified_map


# %%
team_qualified_map = simulate(df_women, 5000000)
end_time = time.time()

total_time = end_time - start_time

# %%
standings = pd.Series(
    {team: np.mean(bool_list) for team, bool_list in team_qualified_map.items()}
)

standings.sort_values(ascending=False).to_csv("predicted_probs_women.csv")

print(total_time)
