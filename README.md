# US_Wildfires

## Data
Data was downloaded from kaggle: https://www.kaggle.com/rtatman/188-million-us-wildfires. As the database file is too big to upload on github, you can download it from kaggle.

Data contains geolocalized data for 1.88 milions of wildfires occuring in the United States from 1992 until 2015. All of the incidents have specified time span of the fire, size and number of id's from different wildfire agencies

## Project info
### Analysis
Data was analysed to gain insights about localization, yearly number, mean size etc. of fires to find areas most affected by wildfires and trends in fires properties. Number of visualizations was prepared for different information to show the data in more user-friendly way

### Classification
Second part of the project was to classify fires based on size, location, length and date of discovery into fire causes classes. Number of different classifier were trained and evaluated to find the best one based on accuracy and F1 score. Best classifier was further evaluated based on precission and recalled score. High recall and sufficient precission were achieved. Classifier can be used to check whether cause of fire is suspicious and requires further investigation


### Files in repository
US-Wildfires-Analysis-Visualisation: file contains analysis and static visualisations of data
US-Wildfires-Modeling: file contains classifier
Visualization - 2014 CA Fires: animated visualization of Wildfires in CA in 2014
Visualization - Fires Yearly: animated visualization of all wildfires per year

scripts: additional functions
vector map folder: vector map of US divided by states
plot_2014.rar: gif build from Visualization - 2014 CA Fires
plot_yearly: gif build from Visualization - Fires Yearly
