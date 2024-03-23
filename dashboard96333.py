import dash
import pickle
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
import plotly.graph_objects as go
import base64
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# define global variables
y_pred_list = []
y_pred_2019 = []
model = None  # Define model as a global variable


# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
white_text_style = {'color': 'white'}

#Load data
#Raw Data
df_together = pd.read_csv("df_together.csv", index_col=0, parse_dates=True) #contains every meteorological data from 201, 2018 and 2019
columns = df_together.columns.tolist()
start_date = df_together.index.min()
end_date = df_together.index.max()

# Set a variable with the cut-off date for the 2019 set
test_cutoff_date = '2019-01-01'

# Split the dataset into training and test sets
df_data_20172018 = df_together.loc[df_together.index < test_cutoff_date] #dataset with values from 2017 and 2018
df_2019_all = df_together.loc[df_together.index >= test_cutoff_date] #dataset with values from 2019 (which are going to predict)

df_data = df_data_20172018.dropna() #Clean the first dataframe

df_data_features = df_data.copy()
df_data_features = df_data.drop("Power (kW)", axis=1).reset_index(drop=True)

df_2019_real = pd.read_csv('df_2019_real.csv')
df_2019_real['Date'] = pd.to_datetime(df_2019_real['Date'])
y=df_2019_real['Power (kW)'].values

df_meteo_2019 = df_2019_all.drop('Power (kW)', axis=1)
df_meteo_2019 = df_meteo_2019.iloc[1:]

#Variabeles initialization
X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None

fig2 = px.line(df_2019_real, x='Date', y='Power (kW)')

#auxiliary functions
def generate_table(dataframe, max_rows=10):

    # Create table headers
    table_headers = [html.Th(col) for col in dataframe.columns]
    
    # Create table rows
    table_rows = []
    for i in range(min(len(dataframe), max_rows)):
        row_data = [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]
        table_rows.append(html.Tr([html.Td(i)] + row_data))  # Include index as the first column
    
    # Construct the table
    table = html.Table([
        html.Thead(html.Tr(table_headers)),
        html.Tbody(table_rows)
    ], style={'borderCollapse': 'collapse', 'borderSpacing': '0', 'width': '100%', 'border': '1px solid #ddd',
              'fontFamily': 'Arial, sans-serif', 'fontSize': '14px'})
    
    return table

def generate_graph(df, columns, start_date, end_date):
  
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Create traces for each column
    traces = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    
    # Configure the layout
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    
    return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(style={'font-family': 'Arial, Helvetica, sans-serif', 'backgroundColor': '#f0f0f0'}, children=[
    html.Div([
        html.Img(src="https://eduportugal.eu/wp-content/uploads/2022/03/Logo_Inst-Tecnico.png", style={'height': '60px', 'width': '100px', 'position': 'absolute', 'top': '10px', 'right': '10px'}),  # Add your image URL here
        html.H1('IST South Tower Energy Dashboard ist196333', style={'margin-right': '60px'}),
    ], style={'display': 'inline-block', 'verticalAlign': 'top'}),
    dcc.Tabs(id='tabs', value='tab-3', children=[
        dcc.Tab(label='Raw Data', value='tab-3', children=[
            html.Div([
                html.H2("Raw Data", style={'backgroundColor': '#ADD8E6', 'padding': '10px', 'border-radius': '5px'}),
                html.P('In order to see the raw data, dont forget to select the data range and also the variables you may want to visualize. After, lock the variables so that the graph is constructed. After that, click clear to build another one, as you wish.'),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in ['2017-2018', '2019', 'All Years']],
                    value='All Years',
                ),
                dcc.Dropdown(
                    id='feature-dropdown-raw',
                    options=[{'label': col, 'value': col} for col in df_together.columns],
                    value=[df_together.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=df_together.index.min(),
                    max_date_allowed=df_together.index.max(),
                    start_date=df_together.index.min(),
                    end_date=df_together.index.max()
                ),
                html.Div([
                    html.Button('Lock Variables', id='lock-button'),  # Add the Lock Variables button here
                    html.Button('Clear Graph', id='clear-button'),  # Add the Clear Graph button here
                    #html.Button('Check Shapes and Contents', id='button-check-shape')  # Add the Check Shapes and Contents button here
                ]),
                dcc.Graph(id='graph'),
            ])
        ]),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2', children=[
            html.Div([
                html.H2("Exploratory Data Analysis", style={'backgroundColor': '#ADD8E6', 'padding': '10px', 'border-radius': '5px'}),
                html.P('Choose the variables to see their interrealtionship in a scatter plot. In case you want to check for outliers, choose one variable at the bottom of the page to create the box plot graphic.'),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in df_together.columns],
                    value=df_data.columns[0]
                ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in df_together.columns],
                    value=df_data.columns[1]
                ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in df_together.columns],
                    value=df_together.columns[1]
                ),
                dcc.Graph(id='box-plot')
            ])
        ]),
        dcc.Tab(label='Feature Selection', value='tab-4', children=[
            html.Div([
                html.H2("Feature Selection", style={'backgroundColor': '#ADD8E6', 'padding': '10px', 'border-radius': '5px'}),
                html.P('Select from the available features the ones you want to train the model. A message "Variables locked successfully!" should appear next to the Lock Variables button.'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df_data_features.columns if col != 'index'],
                    value=[df_data_features.columns[0]],
                    multi=True
                ),
                html.Div(id='feature-table-div'),
                html.Button('Lock Variables', id='split-button'),
                html.Div(id='split-values'),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="y-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-2019-values", style=white_text_style)
                ]),
            ])
        ]),
        dcc.Tab(label='Regression Models', value='tab-5', children=[
            html.Div([
                html.H2("Regression Models", style={'backgroundColor': '#ADD8E6', 'padding': '10px', 'border-radius': '5px'}),
                html.P('Please select the type of model you would like to train, based on the chosen features. After making your selection, click the button to generate the model and visualize the graph.'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Linear Regression', 'value': 'linear'},
                        {'label': 'Random Forests', 'value': 'random_forests'},
                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                        {'label': 'Decision Tree Regressor', 'value': 'decision_trees'}
                    ],
                    value='linear'
                ),
                html.Button('Train Model', id='train-model-button'),
            ]),
            html.Div([
                html.H2(""),
                dcc.Loading(
                    id="loading-1",
                    children=[html.Div([dcc.Graph(id="lr-graph")])]
                )
            ]),
        ]),
        dcc.Tab(label='Model Deployment', value='tab-6', children=[
            html.Div([
                html.H2('Model Deployment', style={'backgroundColor': '#ADD8E6', 'padding': '10px', 'border-radius': '5px'}),
                html.P('Press Run Model Button at the bottom left corner to generate the predictions for 2019 and the results graphically. You can also see the error metrics, which are expressively high due to the feature available for selection'),
                dcc.Graph(id='time-series-plot', figure=fig2),
                dcc.Graph(id='scatter-plot-real-predicted'),  # Add this line to define the scatter plot
                html.Button('Run Model', id='button_model'),
                html.Div(id='model-performance-table')
            ])
        ]),
    ]),
    html.Div(id='tabs-content')
])



# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('feature1', 'value'),
     Input('feature2', 'value')]
)
def update_scatter_plot(feature1, feature2):
    fig = {
        'data': [{
            'x': df_together[feature1],
            'y': df_together[feature2],
            'mode': 'markers'
        }],
        'layout': {
            'title': f'{feature1} vs {feature2}',
            'xaxis': {'title': feature1},
            'yaxis': {'title': feature2},
        }
    }
    return fig

# Callback to update box plot
@app.callback(
    Output('box-plot', 'figure'),
    [Input('feature-boxplot', 'value')]
)
def update_box_plot(feature_boxplot):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_together[feature_boxplot], name=feature_boxplot))
    fig.update_layout(title=f"Box Plot for {feature_boxplot}", title_x=0.5)
    return fig

# Callback to update raw data graph
@app.callback(
    Output('graph', 'figure'),
    [Input('lock-button', 'n_clicks'),
     Input('clear-button', 'n_clicks')],
    [State('column-dropdown', 'value'),
     State('feature-dropdown-raw', 'value'),
     State('date-picker', 'start_date'),
     State('date-picker', 'end_date')]
)
def update_figure(lock_clicks, clear_clicks, selected_column, selected_features, start_date, end_date):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'lock-button':
            # Filter DataFrame based on selected date range and columns
            filtered_df = df_together.loc[start_date:end_date, selected_features]
            data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
            layout = {'title': ', '.join(selected_features), 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Values'}}
            return {'data': data, 'layout': layout}
        elif button_id == 'clear-button':
            return {'data': [], 'layout': {}}
    return {'data': [], 'layout': {}}

@app.callback(
    [Output('date-picker', 'min_date_allowed'),
     Output('date-picker', 'max_date_allowed'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    [Input('column-dropdown', 'value')]
)
def update_date_picker_options(selected_timestamp):
    if selected_timestamp == '2017-2018':
        min_allowed = df_together.index[df_together.index.year == 2017].min()
        max_allowed = df_together.index[df_together.index.year == 2018].max()
        start_date = min_allowed
        end_date = max_allowed
    elif selected_timestamp == '2019':
        min_allowed = df_together.index[df_together.index.year == 2019].min()
        max_allowed = df_together.index[df_together.index.year == 2019].max()
        start_date = min_allowed
        end_date = max_allowed
    else:  # All Years
        min_allowed = df_together.index.min()
        max_allowed = df_together.index.max()
        start_date = min_allowed
        end_date = max_allowed
    
    return min_allowed, max_allowed, start_date, end_date

# Callback to update feature table
@app.callback(
    Output('feature-table-div', 'children'),
    Input('feature-dropdown', 'value')
)
def update_feature_table(selected_features):
    if selected_features:
        global df_model
        df_model = df_data_features[selected_features]
        table = generate_table(df_model)
        return table
    else:
        return html.Div()

# Callback to lock selected features
@app.callback(
    [Output('split-values', 'children'),
     Output('x-values', 'children'),
     Output('y-values', 'children'),
     Output('x-2019-values', 'children')],
    [Input('split-button', 'n_clicks')],
    [State('feature-dropdown', 'value')]
)
def lock_features(n_clicks, selected_features):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'split-button' and selected_features:
            global X, Y, X_train, X_test, y_train, y_test, X_2019
            X = df_model
            Y = df_data['Power (kW)']
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            X_2019 = df_meteo_2019[selected_features]
            print('é x-2019999:', X_2019)
            return (
                html.Div([
                    html.P("Features locked successfully!"),
                    html.P("Selected Features:"),
                    html.Pre(selected_features),
                ]),
                html.Div([
                    html.P("X Values (Training Set):"),
                    html.Pre(X_train.to_string())
                ]),
                html.Div([
                    html.P("Y Values (Training Set):"),
                    html.Pre(y_train.to_string())
                ]),
                html.Div([
                    html.P("X Values (Test Set):"),
                    html.Pre(X_test.to_string())
                ])
            )
    raise PreventUpdate

# Callback to train and predict using selected model
@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def train_and_predict(n_clicks, model_type):
    global y_pred_list, y_pred_2019, model  # access global variables

    if n_clicks is None:
        return dash.no_update 
    else:
        if model_type == 'linear':
            # Linear Regression
            model = LinearRegression()
        elif model_type == 'random_forests':
            # Random Forests
            parameters = {'bootstrap': True,
                          'min_samples_leaf': 3,
                          'n_estimators': 200, 
                          'min_samples_split': 15,
                          'max_features': 'sqrt',
                          'max_depth': 20,
                          'max_leaf_nodes': None}
            model = RandomForestRegressor(**parameters)
        elif model_type == 'bootstrapping':
            # Bootstrapping
            model = BaggingRegressor()
        elif model_type == 'decision_trees':
            # Decision Trees
            model = DecisionTreeRegressor()
        elif model_type == 'gradient_boosting':
            # Gradient Boosting
            model = GradientBoostingRegressor()
        
        # Train the model using the training sets
        model.fit(X_train, y_train)

        # Save the trained model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
            file.close()

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_list.append(y_pred)
        
        y_pred_2019 = model.predict(X_2019)
        
        # Print statements for debugging
        print("Shape of X_2019:", X_2019.shape)
        print("Shape of y_pred_2019:", y_pred_2019.shape)
        
        # Generate scatter plot of predicted vs actual values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers'))
        fig.update_layout(title=f'{model_type.capitalize()} Predictions')
        
        return fig

# Callback to run the model and display results
@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('scatter-plot-real-predicted', 'figure'),
     Output('model-performance-table', 'children')],
    Input('button_model', 'n_clicks')
)
def run_model(n_clicks):
    global y_pred_2019, model  # access global variables
    if n_clicks is None:
        raise PreventUpdate
    else:
                
        # Make predictions
        y_pred_2019 = model.predict(X_2019)
        
        # Plot of Real vs. Predicted Power - time series
        fig = go.Figure(layout=go.Layout(title='Real vs Predicted Power Consumption'))
        fig.add_scatter(x=df_2019_real['Date'].values, y=df_2019_real['Power (kW)'].values, name='Real Power')
        fig.add_scatter(x=df_2019_real['Date'].values, y=y_pred_2019, name='Predicted Power') 

        # Plot of Real vs. Predicted Power - scatter plot
        scatter_plot_fig = go.Figure()
        scatter_plot_fig.add_trace(go.Scatter(x=df_2019_real['Power (kW)'], y=y_pred_2019, mode='markers'))
        scatter_plot_fig.update_layout(
            title='Real vs Predicted Power Consumption',
            xaxis_title='Real Power',
            yaxis_title='Predicted Power'
        )

        # Calculate model performance metrics
        MAE = mean_absolute_error(df_2019_real['Power (kW)'].values, y_pred_2019)
        MBE = np.mean(df_2019_real['Power (kW)'].values - y_pred_2019)
        MSE = mean_squared_error(df_2019_real['Power (kW)'].values, y_pred_2019)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(df_2019_real['Power (kW)'].values)
        nmbe = MBE / np.mean(df_2019_real['Power (kW)'].values)

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
        return fig, scatter_plot_fig, table


# Run the app
if __name__ == '__main__':
    #app.run_server(debug=True, host='127.0.0.1', port=8050)
    app.run_server()