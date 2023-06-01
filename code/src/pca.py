import math
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE

def pca_2d_plot(file_name : str = "save-100iter-no_dup_acc.csv"):
    pca = PCA(n_components=2)

    db = pd.read_csv(file_name)

    result = pca.fit_transform(db.iloc[:,:-3])

    result_df = pd.DataFrame(data = result
                , columns = ['principal component 1', 'principal component 2'])

    result_df['y'] = db['fitness']

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=result_df,
        legend="full",
        alpha=0.3
    )
    plt.show()

def pca_plot(mission_name,file_name : str = "save-100iter-no_dup_acc.csv",  feature = "SENSORY", is3D = False, nbest = False):
        title= f"{feature} - {mission_name} "
        AX1 = "PC1"
        AX2 = "PC2"
        AX3 = "PC3"
        pca = PCA(n_components=3)
        db = pd.read_csv(file_name)

        if nbest:
            result = pca.fit_transform(db.iloc[:,:-4])
        else:
            result = pca.fit_transform(db.iloc[:,:-3])

        result_df = pd.DataFrame(data = result
                    , columns = ['1', '2','3'])

        result_df['y'] = 1
        if nbest:
            db = db.nlargest(100,"fit")
            result_df['y'] = db['fit']

        cmap = ListedColormap(sns.color_palette("rocket_r").as_hex())

        if is3D:
            fig = plt.figure(figsize=(16,10))
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)


            sc = ax.scatter(result_df['1'], result_df['2'], result_df['3'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
            ax.set_xlabel(AX1)
            ax.set_ylabel(AX2)
            ax.set_zlabel(AX3)

            plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
        else:
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))

            # XY plane
            im = axs[0, 0].scatter(result_df['1'], result_df['2'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
            axs[0, 0].set_xlabel(AX1)
            axs[0, 0].set_ylabel(AX2)

            # XZ plane
            im = axs[0, 1].scatter(result_df['1'], result_df['3'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
            axs[0, 1].set_xlabel(AX1)
            axs[0, 1].set_ylabel(AX3)

            # YX plane
            im = axs[1, 0].scatter(result_df['2'], result_df['1'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
            axs[1, 0].set_xlabel(AX2)
            axs[1, 0].set_ylabel(AX1)

            # YZ plane
            im = axs[1, 1].scatter(result_df['2'], result_df['3'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
            axs[1, 1].set_xlabel(AX2)
            axs[1, 1].set_ylabel(AX3)

            # Add a giant colorbar to the right of all subplots
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
            fig.suptitle(f'2D Scatter Plots from All Sides of 3D PCA Plot {title} ', fontsize=16)

        # plt.show()
        plt.savefig(f'{feature}_{mission_name}.png')

def pca_illuminate(index, mission:int,mission_name,fitness,file_name : str = "save-100iter-no_dup_acc.csv",  feature = "POS", is3D = False,times = 1):
        from matplotlib.colors import ListedColormap, Normalize

        mis = ["AAC","COVERAGE","SHELTER","HOMING"]
        title= f"{feature} - {mission_name} "
        AX1 = "PC1"
        AX2 = "PC2"
        AX3 = "PC3"
        pca = PCA(n_components=3)
        db = pd.read_csv(file_name)

        result = pca.fit_transform(db.iloc[:,:-3])

        result_df = pd.DataFrame(data = result
                    , columns = ['1', '2','3'])

        # result_df['y'] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").replace("'","").split(',')[mission]))
        result_df['y'] = db['fitness']
        result_df['y'] = result_df['y'].apply(lambda x: x if x > 210 else 210)

        # result_df['y'] = result_df.apply(lambda x: 300 if x.name == index else 0, axis=1)


        # Create the color map using the normalization object and the color palette
        # Create the color map using the normalized values
        cmap = ListedColormap(sns.color_palette("viridis").as_hex())
        


        # Plot the data
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        # XY plane
        boldness = 2
        im = axs[0, 0].scatter(result_df['1'], result_df['2'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
        axs[0, 0].scatter(result_df.loc[index, '1'], result_df.loc[index, '2'], s=100, c='black', marker='x',linewidths=boldness)
        axs[0, 0].set_xlabel(AX1)
        axs[0, 0].set_ylabel(AX2)

        # XZ plane
        im = axs[0, 1].scatter(result_df['1'], result_df['3'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
        axs[0, 1].scatter(result_df.loc[index, '1'], result_df.loc[index, '3'], s=100, c='black', marker='x',linewidths=boldness)
        axs[0, 1].set_xlabel(AX1)
        axs[0, 1].set_ylabel(AX3)

        # YX plane
        im = axs[1, 0].scatter(result_df['2'], result_df['1'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
        axs[1, 0].scatter(result_df.loc[index, '2'], result_df.loc[index, '1'], s=100, c='black', marker='x',linewidths=boldness)
        axs[1, 0].set_xlabel(AX2)
        axs[1, 0].set_ylabel(AX1)

        # YZ plane
        im = axs[1, 1].scatter(result_df['2'], result_df['3'], s=40, c=result_df['y'], marker='o', cmap=cmap, alpha=1)
        axs[1, 1].scatter(result_df.loc[index, '2'], result_df.loc[index, '3'], s=100, c='black', marker='x',linewidths=boldness)
        axs[1, 1].set_xlabel(AX2)
        axs[1, 1].set_ylabel(AX3)

        # Add a giant colorbar to the right of all subplots
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
        fig.suptitle(f'{title} - best score : {fitness} ', fontsize=16)


        # plt.show()
        plt.savefig(f'{times}_{mission_name}_illuminate.png')

def pca_variance(file_name : str = "save-100iter-no_dup_acc.csv"):
    db = pd.read_csv(file_name)
    pca = PCA(10)
    result = pca.fit_transform(db.iloc[:,:-4])

    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = [f'PC{str(x)}' for x in range(0,len(per_var))]    

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label = labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title("Explained Variance Positional")
    plt.show()

def bar_plot(file_name,type = "SENSORY", evaluated = False):
    if type == "SENSORY":
        names = ["mean linear velocity","sdv linear velocity","mean angular velocity","sdv angular velocity","mean dist to wall","sdv dist to wall","mean dist to robot","sdv dist to robot","mean dist to min","sdv dist to min","mean light","sdv light","mean color","sdv color"]
    else:
        names=[
            "mean dist to black 1","sdv dist to black 1",
            "mean dist to black 2","sdv dist to black 2",
            "mean dist to black 3","sdv dist to black 3",
            "mean dist to black 4","sdv dist to black 4",
            "mean dist to black 5","sdv dist to black 5",
            "mean dist to black 6","sdv dist to black 6",
            "mean dist to black 7","sdv dist to black 7",
            "mean dist to black 8","sdv dist to black 8",
            "mean dist to black 9","sdv dist to black 9",
            "mean dist to black 10","sdv dist to black 10",
            "mean dist to black 11","sdv dist to black 11",
            "mean dist to black 12","sdv dist to black 12",
            "mean dist to black 13","sdv dist to black 13",
            "mean dist to black 14","sdv dist to black 14",
            "mean dist to black 15","sdv dist to black 15",
            "mean dist to black 16","sdv dist to black 16",
            "mean dist to black 17","sdv dist to black 17",
            "mean dist to black 18","sdv dist to black 18",
            "mean dist to black 19","sdv dist to black 19",
            "mean dist to black 20","sdv dist to black 20",
            "mean dist to white 1","sdv dist to white 1",
            "mean dist to white 2","sdv dist to white 2",
            "mean dist to white 3","sdv dist to white 3",
            "mean dist to white 4","sdv dist to white 4",
            "mean dist to white 5","sdv dist to white 5",
            "mean dist to white 6","sdv dist to white 6",
            "mean dist to white 7","sdv dist to white 7",
            "mean dist to white 8","sdv dist to white 8",
            "mean dist to white 9","sdv dist to white 9",
            "mean dist to white 10","sdv dist to white 10",
            "mean dist to white 11","sdv dist to white 11",
            "mean dist to white 12","sdv dist to white 12",
            "mean dist to white 13","sdv dist to white 13",
            "mean dist to white 14","sdv dist to white 14",
            "mean dist to white 15","sdv dist to white 15",
            "mean dist to white 16","sdv dist to white 16",
            "mean dist to white 17","sdv dist to white 17",
            "mean dist to white 18","sdv dist to white 18",
            "mean dist to white 19","sdv dist to white 19",
            "mean dist to white 20","sdv dist to white 20",
            "mean closest 1","sdv closest 1",
            "mean closest 2","sdv closest 2",
            "mean closest 3","sdv closest 3",
            "mean closest 4","sdv closest 4",
            "mean closest 5","sdv closest 5",
            "mean closest 6","sdv closest 6",
            "mean closest 7","sdv closest 7",
            "mean closest 8","sdv closest 8",
            "mean closest 9","sdv closest 9",
            "mean closest 10","sdv closest 10",
            "mean closest 11","sdv closest 11",
            "mean closest 12","sdv closest 12",
            "mean closest 13","sdv closest 13",
            "mean closest 14","sdv closest 14",
            "mean closest 15","sdv closest 15",
            "mean closest 16","sdv closest 16",
            "mean closest 17","sdv closest 17",
            "mean closest 18","sdv closest 18",
            "mean closest 19","sdv closest 19",
            "mean closest 20","sdv closest 20",
        ]
    print(len(names))
    index = 4 if evaluated else 3
    db = pd.read_csv(file_name)
    pca = PCA(10)
    result = pca.fit_transform(db.iloc[:,:-3])
    for i in range(6): 
        loading_score = pd.Series(pca.components_[i],index = db.columns[:-3])
        sorted_loading = loading_score.abs().sort_values(ascending=False)
        top10 = sorted_loading[:10].index.values
        scores = loading_score[top10].to_list()
        scores = [abs(score) for score in scores]
        top10 = [names[int(x)-1] for x in top10]
        colors = ['g' if 'sdv' in variable else 'b' for variable in top10]
        print(top10)
        plt.figure(figsize=(8,8))
        plt.bar(top10, scores,color=colors)
        plt.title(f'Importance of features for PC {i}')
        plt.ylabel('importance [%]')
        plt.xticks(rotation=30, ha='right')
        plt.savefig(f'PC{i}_{type}_barplot.png')


def loading_scores(file_name : str = "save-100iter-no_dup_acc.csv"):
    db = pd.read_csv(file_name)
    pca = PCA(10)
    result = pca.fit_transform(db.iloc[:,:-4])
    loading_score = pd.Series(pca.components_[0],index = db.columns[:-4])
    sorted_loading = loading_score.abs().sort_values(ascending=False)
    top10 = sorted_loading[:10].index.values
    print(loading_score[top10])
    

def get_best_pfsm(amount, type,file_name,output="best_pfsm.txt"):
    db = pd.read_csv(file_name)
    db[type] = db['fitness']
    # db[type] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").replace("'","").split(',')[0]))
    # db['COVERAGE'] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").split(',')[1]))
    # db['HOMING'] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").split(',')[2]))
    # db['SHELTER'] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").split(',')[3]))
    print(db.head())

    best_pfsm = []
    test = np.array(db.geno.tolist())
    print("unique is ", len(np.unique(test)))
    with open(output,"w") as f:
        f.write(f"{type}\n")
        best = db.nlargest(amount,type)
        indexes = best.index.tolist()
        scores = best.fitness.tolist()
        pfsms = best.geno.tolist()
        for i in range(len(pfsms)):
            f.write(f"{indexes[i]} - score {scores[i]}\n")
            f.write(f"{pfsms[i]}\n")
        # for mission in missions:
            # print(best.geno.head())
            # best_pfsm.append(pfsms)
    # print(best_pfsm)

def show_best(file,type = "AAC"):
    db = pd.read_csv(file)
    db[type] = db['fitness']
    # db[type] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").replace("'","").split(',')[0]))
    best = db.nlargest(1,type)
    indexes = best.index.tolist()
    fitness = best[type].tolist()
    for i in range(len(best)):
        pca_illuminate(indexes[i],0,type,fitness[i],file,times=i,is3D=True)


def tsne(file,type, set):
    db = pd.read_csv(file)
    X = db.iloc[:,:-4]
    X_tsne = TSNE(n_components=2).fit_transform(X)
    tsne_x = X_tsne[:,0]
    tsne_y = X_tsne[:,1]

    # y = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").replace("'","").split(',')[0]))
    y = db['fit']
    fig, ax = plt.subplots()
    scatter = ax.scatter(tsne_x, tsne_y, c=y)

    ax.set_title(f'TSNE {set} {type}')
    fig.colorbar(scatter, shrink=0.6)


    plt.savefig(f"TSNE_{type}.png")

def box_plot_best( n_best= 100):
    files = [
        [
            "/home/laurent/Documents/Polytech/MA2/thesis/sensory/AAC_SENSORY_features.csv",
        ],
        [
            "/home/laurent/Documents/Polytech/MA2/thesis/positional/AAC_POSITIONAL_features.csv",
            # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/new_protocol/AAC_WHITE_ALL_SENSORY_features.csv",
            
            # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/new_protocol/FORBID_ALL_SENSORY_features.csv"
        ]
    ]

    names = [
        [
            "AAC"
        ],
        [
            "AAC",
            # "WHITE",
            # "FORBID",
        ]
    ]
    titles = ["POS","SENSORY"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle("AGGREGATION", fontsize=16)
    best = []
    for i in range(len(names)):
        best_temp = []
        for j in range(len(names[i])):
            db = pd.read_csv(files[i][j])
            b = db.nlargest(n_best,"fitness")
            best_temp.append(b.fitness)
        # best.append(best_temp)
        im = axs[i].boxplot(best_temp,notch=True)
        axs[i].set_xticklabels(names[i],rotation=45)
        axs[i].title.set_text(titles[i])

    # fig = plt.figure(figsize =(10, 7))
    # plt.boxplot(best.fit)
    plt.savefig(f"box_plot_{n_best}")


    # file = "/home/laurent/Documents/Polytech/MA2/thesis/sensory/new_protocol/FORBID_ALL_SENSORY_features.csv"
    # db = pd.read_csv(file)
    # db['fit'] = db['fit'].apply(lambda x: float(x.replace("[","").replace("]","").replace("'","").split(',')[0]))
    # db.to_csv("/home/laurent/Documents/Polytech/MA2/thesis/positional/new_protocol/AAC_BLACK_ALL_POS_features.csv")

def most_important_feature(file,mission, type = "SENSORY"):
    if type == "SENSORY":
        names = ["mean linear velocity",
                 "sdv linear velocity",
                 "mean angular velocity",
                 "sdv angular velocity",
                 "mean dist to wall",
                 "sdv dist to wall",
                 "mean dist to other robot",
                 "sdv dist to other robot",
                 "mean dist to closest robot",
                 "sdv dist to closest robot",
                 "mean light",
                 "sdv light",
                 "mean color",
                 "sdv color"]
    else:
        names=[
            "mean dist to black 1","sdv dist to black 1",
            "mean dist to black 2","sdv dist to black 2",
            "mean dist to black 3","sdv dist to black 3",
            "mean dist to black 4","sdv dist to black 4",
            "mean dist to black 5","sdv dist to black 5",
            "mean dist to black 6","sdv dist to black 6",
            "mean dist to black 7","sdv dist to black 7",
            "mean dist to black 8","sdv dist to black 8",
            "mean dist to black 9","sdv dist to black 9",
            "mean dist to black 10","sdv dist to black 10",
            "mean dist to black 11","sdv dist to black 11",
            "mean dist to black 12","sdv dist to black 12",
            "mean dist to black 13","sdv dist to black 13",
            "mean dist to black 14","sdv dist to black 14",
            "mean dist to black 15","sdv dist to black 15",
            "mean dist to black 16","sdv dist to black 16",
            "mean dist to black 17","sdv dist to black 17",
            "mean dist to black 18","sdv dist to black 18",
            "mean dist to black 19","sdv dist to black 19",
            "mean dist to black 20","sdv dist to black 20",
            "mean dist to white 1","sdv dist to white 1",
            "mean dist to white 2","sdv dist to white 2",
            "mean dist to white 3","sdv dist to white 3",
            "mean dist to white 4","sdv dist to white 4",
            "mean dist to white 5","sdv dist to white 5",
            "mean dist to white 6","sdv dist to white 6",
            "mean dist to white 7","sdv dist to white 7",
            "mean dist to white 8","sdv dist to white 8",
            "mean dist to white 9","sdv dist to white 9",
            "mean dist to white 10","sdv dist to white 10",
            "mean dist to white 11","sdv dist to white 11",
            "mean dist to white 12","sdv dist to white 12",
            "mean dist to white 13","sdv dist to white 13",
            "mean dist to white 14","sdv dist to white 14",
            "mean dist to white 15","sdv dist to white 15",
            "mean dist to white 16","sdv dist to white 16",
            "mean dist to white 17","sdv dist to white 17",
            "mean dist to white 18","sdv dist to white 18",
            "mean dist to white 19","sdv dist to white 19",
            "mean dist to white 20","sdv dist to white 20",
            "mean dist to closest robot 1","sdv dist to closest robot 1",
            "mean dist to closest robot 2","sdv dist to closest robot 2",
            "mean dist to closest robot 3","sdv dist to closest robot 3",
            "mean dist to closest robot 4","sdv dist to closest robot 4",
            "mean dist to closest robot 5","sdv dist to closest robot 5",
            "mean dist to closest robot 6","sdv dist to closest robot 6",
            "mean dist to closest robot 7","sdv dist to closest robot 7",
            "mean dist to closest robot 8","sdv dist to closest robot 8",
            "mean dist to closest robot 9","sdv dist to closest robot 9",
            "mean dist to closest robot 10","sdv dist to closest robot 10",
            "mean dist to closest robot 11","sdv dist to closest robot 11",
            "mean dist to closest robot 12","sdv dist to closest robot 12",
            "mean dist to closest robot 13","sdv dist to closest robot 13",
            "mean dist to closest robot 14","sdv dist to closest robot 14",
            "mean dist to closest robot 15","sdv dist to closest robot 15",
            "mean dist to closest robot 16","sdv dist to closest robot 16",
            "mean dist to closest robot 17","sdv dist to closest robot 17",
            "mean dist to closest robot 18","sdv dist to closest robot 18",
            "mean dist to closest robot 19","sdv dist to closest robot 19",
            "mean dist to closest robot 20","sdv dist to closest robot 20",
        ]
    db = pd.read_csv(file,index_col=0)
    # Create a Random Forest classifier
    rf = RandomForestRegressor()

    # Fit the model on the data
    rf.fit(db.iloc[:,:-4], db.fit)

    # Get the feature importances
    importances = rf.feature_importances_

    # Create a DataFrame with feature names and importancesx
    feature_importances = pd.DataFrame({'Feature': names, 'Importance': importances})#db.iloc[:,:-4].columns

    # Sort the features by importance in descending order
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(13, 10))
    plt.barh(feature_importances['Feature'].to_list()[:10], feature_importances['Importance'].to_list()[:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances')
    plt.savefig(f"importance_{type}_{mission}")

def evolution(file,name,algo = "SENSORY",iteration = 6250,population=32):
    mean = {
        "AAC":11726.0,
        "FORAGING":49,
        "HOMING":0.8,
        "SHELTER":6910.5,
        "FORBID":224.238,
    }
    print(mean[name])
    db = pd.read_csv(file).fitness.to_list()
    size = len(db)
    size_to_keep = math.floor(len(db)/population)
    # print("size_to_keep ",size_to_keep*population)
    new_db = db[size-size_to_keep*population:]
    box_plot = []
    index = 0
    best = []
    for i in range(size_to_keep):
        temp = [new_db[i*population+j] for j in range(population) if j < population]
        best.extend(temp)
        if i != 0:
            best.sort(reverse=True)
        box_plot.append(best[:10])
            # box_plot.append(sum(temp)/len(temp))
    # print(len(box_plot))
    # print(len(box_plot[0]))
    # counter = 0
    # jump = 1800
    # while counter < size_to_keep:
    plt.figure(figsize=(30,10))
        # if counter + jump > size_to_keep:
        #     plt.boxplot(box_plot[counter:])
        # else:
        #     plt.boxplot(box_plot[counter:counter+jump])
    plt.boxplot(box_plot)
    plt.title(f"Evolution of the {name} mission for {algo}")
    # x_ticks = np.arange(counter, counter+jump, 100)
    x_ticks = np.arange(0, len(box_plot), 500)
    x_tick_labels = [str(x) for x in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)
    plt.axhline(y=mean[name], color='red', linestyle='-',linewidth=6.0)
    plt.xlabel("iterations")
    # plt.ylim(0, 240)  # Set the y-axis limits to 0 and 1
    plt.ylabel("Fitness value")
    plt.savefig(f"evolution_{name}_{algo}.png")
    # counter += jump


def notched_box_plot():
    file = "/home/laurent/Documents/Polytech/MA2/thesis/box_plot/result.csv"
    db = pd.read_csv(file)

    mission = 'FORBID'
    algo = ["sensory_pseudo","sensory","pos_pseudo","pos","chocolate_pseudo","chocolate"]
    data = [
        np.array(
            db.loc[
                (db['task'] == mission) & (db['algo'] == al), 'fitness'
            ].values
        )
        for al in algo
    ]
    plt.figure(figsize=(16,10))
    for i, (d, a) in enumerate(zip(data, algo), start=1):
    # Set the width of the boxplot
        width = 0.6 if i % 2 != 0 else 0.2
    
    # Plot the boxplot with the adjusted width
        plt.boxplot(d, positions=[i], widths=width, notch=True)

        if i % 2 == 0:
            plt.axvline(x=i + 0.5, color='black')
    plt.xticks(range(1, len(algo) + 1), algo, rotation=45)
    plt.title(f"{mission}")
    plt.savefig(f"{mission}_box_plot.png")
if __name__ == "__main__":

    files = [
        # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_HOMING_SENSORY_features.csv",
        # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_FORBID_SENSORY_features.csv",
        # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_FORAGING_SENSORY_features.csv",
        "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_HOMING_SENSORY_features.csv",
        # "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_AAC_SENSORY_features.csv",
        
    ]
    names = [
        # "HOMING",
        # "FORBID",
        # "FORAGING",
        "SHELTER",
        # "AAC"
    ]
    # for i in range(len(files)):
    #     if names[i] == "AAC":
    #         evolution(files[i],names[i],iteration=3125)
    #     else:
    #         evolution(files[i],names[i])
    # file = "/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_HOMING_SENSORY_features.csv"
    # evolution(file, "HOMING")
    mission = "FORAGING"
    file = f"/home/laurent/Documents/Polytech/MA2/thesis/positional/dump_{mission}_POSITIONAL_features.csv"
    evolution(file, f"{mission}","POS")
    file = f"/home/laurent/Documents/Polytech/MA2/thesis/sensory/dump_{mission}_SENSORY_features.csv"
    evolution(file, f"{mission}")
    # get_best_pfsm(10,"FORBID",file)
    # show_best(file,"FORBID")
    # notched_box_plot()
    # get_best_pfsm(10,"FORBID_POS",file)

    # box_plot_best(n_best=10)
    # most_important_feature(file,"BLACK","POS")
    # pca_variance(file)
    # loading_scores(file)
    # bar_plot(file,"POS",evaluated = True)
    # pca_plot("FORBID",file, feature="SENSORY",nbest = True)
