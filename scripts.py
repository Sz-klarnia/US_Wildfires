#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import date, timedelta
import datetime
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import shapefile
import geopandas as gpd
from bokeh.models import *
from bokeh.plotting import *
from bokeh.io import *
from bokeh.tile_providers import *
from bokeh.palettes import *
from bokeh.transform import *
from bokeh.layouts import *
# preprocessing, model selections, metrics
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score,f1_score

# classificators
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

from sklearn.linear_model import LinearRegression


# In[2]:


def geodf_create_transform(df,long_col,lat_col,to_resize=None,resize=5):
    """
    Function takes a pandas DataFrame (df), names of columns with Latitude and Longitude properties,
    name of a columns to resize, if neceseary and resizing scale. It returns a GeoDataFrame with latitudes and longitudes 
    translated into pseudo Mercator web projection, used by number of online maps, such as OpenStreetMap
    """
    # creating geodataframe from given df, geometry from long and lat cols
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[long_col], df[lat_col]),crs=("EPSG:4326"))
    
    # changing geographical projection
    gdf = gdf.to_crs("EPSG:3857")
    
    # separating x_coordinates from geometry points
    gdf["x_crds"] = [x for x in gdf[long_col]]
    for i in range(len(gdf.geometry)):
        gdf.x_crds[i] = list(gdf.geometry[i].coords)[0][0]
        
    # separating y_coordinates from geometry points
    gdf["y_crds"] = [y for y in gdf[lat_col]]
    for i in range(len(gdf.geometry)):
        gdf.y_crds[i] = list(gdf.geometry[i].coords)[0][1]
    
    # resizing column if neceseary
    if to_resize != None:
        gdf["resized"] = gdf[to_resize] / resize
        
    # return transformed dataframe
    return gdf


# In[3]:


def get_max_min(x_max,x_min,y_max,y_min):
    """Function takes maximal and minimal geographical coordinates and translates them into pseudo Mercator points
    for the purpose of specyfing the maps' range"""
    # list of x and y coordinates
    x_max_min = [x_max,x_min]
    y_max_min = [y_max, y_min]
    
    # creating shapely points
    extremes = gpd.points_from_xy(x_max_min,y_max_min)
    
    # creating geoseries
    extremes = gpd.GeoSeries(extremes,crs=("EPSG:4326"))
    
    # reprojecting points to pseudo mercator
    extremes = extremes.to_crs("EPSG:3857")
    
    # defining ranges of x an y
    x_range = sorted([list(extremes[0].coords)[0][0], list(extremes[1].coords)[0][0]])
    y_range = sorted([list(extremes[0].coords)[0][1], list(extremes[1].coords)[0][1]])
    
    return x_range, y_range


# In[4]:


def plot_bubble_map(df,source,radius_col,hover_tuples,x_range,y_range,title=None,leg_label=None):
    
    """ Ploting a stationary bubble map to display some phenomenon geographical distribution
    Function requires:
    Pandas or GeoPandas DataFrame with data
    Bokeh specific ColumnDataSource
    List of HoverTuples for creating a Hover Tool for map
    x and y range of the map to display
    Optional:
    a title to be displayed
    Legend label for circle plot
    
    Function automatically plots and displays a map"""
    
    # specyfying map provider
    tile_provider = get_provider(OSM)

    # plotting figure
    plot=figure(
        title=title,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        x_range=x_range,
        y_range=y_range,
        x_axis_type='mercator',
        y_axis_type='mercator'
        )

    # switching of grid
    plot.grid.visible=True
    
    # plotting circles representing fires
    c = plot.circle("x_crds", "y_crds",source=source, size=1,radius=radius_col, color="red",alpha=0.5,legend=leg_label)
    
    # creating HoverTool to display fire properties when mouse hovers above points
    circle_hover = HoverTool(tooltips=[x for x in hover_tuples],mode='mouse',point_policy='follow_mouse',renderers=[c])
    # rendering points, adding them to plot
    circle_hover.renderers.append(c)
    plot.tools.append(circle_hover)
    
    # adding open street map
    map_=plot.add_tile(tile_provider)
    map_.level='underlay'
    
    # switching of axes
    plot.xaxis.visible = False
    plot.yaxis.visible=False
    
    show(plot)
    output_notebook()


# In[5]:


def plot_bubble_map_for_animation(df,source,radius_col,hover_tuples,x_range,y_range,title=None, leg_label=None):
    """ Version of plot_bubble_map function that allows notebook updates in real time, for purpose of creating animated maps.
    Instead of plot_bubble_map function does not automatically plot objects, but creates and returns a plot object. 
    """
    
    # specyfying map provider
    tile_provider = get_provider(OSM)

    # plotting figure
    plot=figure(
        title=title,
        match_aspect=True,
        tools='wheel_zoom,pan,reset,save',
        x_range=x_range,
        y_range=y_range,
        x_axis_type='mercator',
        y_axis_type='mercator'
        )

    # switching of grid
    plot.grid.visible=True
    
    # plotting circles representing fires
    c = plot.circle("x_crds", "y_crds",source=source, size=1,radius=radius_col, color="red",alpha=0.5,legend=leg_label)
    
    # creating HoverTool to display fire properties when mouse hovers above points
    circle_hover = HoverTool(tooltips=[x for x in hover_tuples],mode='mouse',point_policy='follow_mouse',renderers=[c])
    # rendering points, adding them to plot
    circle_hover.renderers.append(c)
    plot.tools.append(circle_hover)
    
    # adding open street map
    map_=plot.add_tile(tile_provider)
    map_.level='underlay'
    
    # switching of axes
    plot.xaxis.visible = False
    plot.yaxis.visible=False
    
    return plot


def find_best_model(X_train, X_test, y_train, y_test, pipelines, param_grids):
    # using grid search cv to find the best model

    # defining variables to store parameters and scores for the best model
    best_model = ""
    best_params = ""
    score_f1 = 0
    acc_score = 0

    # testing in a loop for each pipeline - parameters combination
    for i in range(len(pipelines)):

        # Creating GridSearch object
        grid = GridSearchCV(pipelines[i],
                            param_grid=param_grids[i],
                            refit=True,  # refitting best model
                            cv=5,  # nr of crossvalidations
                            verbose=3,  # provides information during the learning process
                            n_jobs=3)  # nr of cores used for testing

        # Fitting data
        grid.fit(X_train, y_train)

        # Predicting on test data, scoring
        y_pred = grid.predict(X_test)
        test_f1 = f1_score(y_test, y_pred,average="micro")
        test_acc = accuracy_score(y_test, y_pred)

        # Checking if model is better than current best, declaring new values if it is
        if test_f1 > score_f1:
            acc_score = test_acc
            score_f1 = test_f1
            best_model = pipelines[i]
            best_params = grid.best_params_

    # printing best model
    print(
        f"Best model: {best_model} with {best_params} parameters. F1 score = {score_f1} and Accuracy score = {acc_score}")
