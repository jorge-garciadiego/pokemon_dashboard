import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output
import dash_bootstrap_components as dbc

chart_layout_font = {'color': 'rgb(255, 204, 0)'}

'''
Data source from Database *****
'''
# hostname = ''
# database = ''
# username = ''
# pwd = 'password'
# port_id = 5432
# conn = None
# cur = None


# # //[USERNAME]:[PASSWORD]@[DB_HOST]:[PORT]/[DB_NAME]
# engine = sqlalchemy.create_engine(f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}")
    
# query_all_pokemon = '''
#     SELECT pk.pokedex_number, pk.name, pkt.pokemon_type AS "Type1",
#     pkt2.pokemon_type AS "Type2", pk.attack, pk.defense, pk.experience_growth,
#     pk.base_egg, pk.base_happiness, pk.base_total, pk.capture_rate,
#     pk.height_m, pk.weight_kg, pk.generation, pk.is_legendary, pk.hp,
#     pk.sp_attack, pk.sp_defense, pk.speed, pk.abilities
#     FROM pokemon as pk
#     INNER JOIN pokemon_types as pkt ON pk.type1_id = pkt.id
#     INNER JOIN pokemon_types as pkt2 ON pk.type2_id = pkt2.id;
# '''
    
# pokemon_df = pd.read_sql_query(sql=query_all_pokemon, con=engine)
       
'''
Read dataset for Deploy purposes
'''
pokemon_df = pd.read_csv('pokemon_dataset.csv')

print(f"Unique primary types: {pokemon_df['Type1'].nunique()}")
print(f"Unique secundary types: {pokemon_df['Type2'].nunique()}")

# pokemon_df.drop(['japanese_name', 'base_egg_steps', 'percentage_male'],
#                 axis=1, inplace=True)

pokemon_df.rename(str.title, axis='columns', inplace=True)

# A list of the unique types
types = [t for t in list(pokemon_df['Type1'].unique())]
# Replace types values as title format
pokemon_df['Type1'].replace(types, [t.title() for t in types], inplace=True)
pokemon_df['Type2'].replace(types, [t.title() for t in types], inplace=True)

# A feature joining type1 and type 2 into a single type
pokemon_df['Type'] = pokemon_df.apply(
    lambda x: x['Type1'] if x['Type2'] == 'NaN' else f'{x["Type1"]}_{x["Type2"]}', 
    axis=1
)

'''
The Abilities column has a stringified list we convert it back to a list
'''
# To convert a stringified list into a list
from ast import literal_eval
pokemon_df['Abilities'] = pokemon_df.apply(lambda x: literal_eval(x['Abilities']), axis=1)

'''
We create a column with the number of abilities
'''

pokemon_df['N_Abilities'] = pokemon_df.apply(lambda x: len(x['Abilities']), axis=1)

'''
We calculate the BMI for each pokemon
'''
pokemon_df['BMI'] = pokemon_df.apply(lambda x: x['Weight_Kg'] / (x['Height_M']**2), axis=1)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

server = app.server

'''
Bar char with the number of pokemons by Generation
'''
# Create a list of generations
generations = [f'Gen {g}' for g in pokemon_df.Generation.unique().tolist()]

# Count the pokemon by generation
total_by_gen = pokemon_df.groupby('Generation').count()['Type'].tolist()

bar_gen = px.bar(
    x=total_by_gen,
    y=generations,
    orientation='h',
    text=total_by_gen
)

bar_gen.update_layout(barmode='stack', yaxis={'categoryorder':'category descending', 'title': 'Generation'}, xaxis={'title': 'Number of Pokemon'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=chart_layout_font)

'''
A Tree map chart showing the top ten Base_Total pokemon by Generation
'''
# a list of dataframes holding the top ten by generation
top_ten_list =[
    pokemon_df[['Generation', 'Type', 'Name','Attack', 'Defense', 'Speed', 'Capture_Rate', 'Hp', 'Base_Total']]
    .loc[pokemon_df['Generation'] == d].sort_values(by='Base_Total', ascending=False).iloc[:10] for d in pokemon_df['Generation'].unique()
]
# We concatenate the dataframes on the top_ten_list into a single dataframe
top_ten_by_gen = pd.concat(top_ten_list)

# We add a treemap were the size of the squares are the capture_rate and the color scale the attack
tree_chart = px.treemap(
    data_frame=top_ten_by_gen, path=[px.Constant('All'), 'Generation', 'Type', 'Name'], values='Capture_Rate',
    color='Attack',
    color_continuous_scale='RdBu'
)

tree_chart.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

'''
Visualize the Type 1 and 2 occurrences
'''
pokemon_type1 = pokemon_df.groupby('Type1').count()['Type'].reset_index()
pokemon_type2 = pokemon_df.loc[pokemon_df['Type2']!= 'NaN'].groupby('Type2').count()['Type'].reset_index()
types_bar = go.Figure()
types_bar.add_trace(go.Bar(
    y= pokemon_type1.iloc[:,0],
    x= pokemon_type1.iloc[:,1],
    name='Type1',
    orientation='h',
    marker=dict(
        color='rgba(251, 27, 27, 0.6)',
        line=dict(color='rgba(251, 27, 27, 1.0)', width=3)
)))

types_bar.add_trace(go.Bar(
    y= pokemon_type2.iloc[:,0],
    x= pokemon_type2.iloc[:,1],
    name='Type2',
    orientation='h',
    marker=dict(
        color='rgba(255, 255, 255, 0.6)',
        line=dict(color='rgba(255, 255, 255, 1.0)', width=3)
)))

types_bar.update_layout(barmode='stack', font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

'''
Visualization of the top 10 types combinations 
'''
# a list of the top 10 combinations taking out the pokemon without Type2
top10_type_counts = pokemon_df.loc[pokemon_df['Type2']!='NaN']["Type"].value_counts()[:10]
top10_combinations = px.bar(
    x = top10_type_counts,
    y = top10_type_counts.index,
    orientation='h'
    
)

category_order = top10_type_counts.sort_values(ascending=True).index.to_list()

top10_combinations.update_traces(marker_color='rgba(255,222,0,0.6)')
top10_combinations.update_layout(barmode='stack', yaxis={'categoryorder':'array', 'categoryarray':category_order}, font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

'''
The Most common Types of Legendary Pokemon
'''

top5_legendary = pokemon_df.loc[pokemon_df['Is_Legendary']==1]['Type1'].value_counts()[:5]

top5_leg_bar = px.bar(
    x=top5_legendary,
    y=top5_legendary.index,
    orientation='h'
)

top5_ordered = top5_legendary.sort_values(ascending=True).index.to_list()
top5_leg_bar.update_traces(marker_color='rgba(255,222,0,0.6)')
top5_leg_bar.update_layout(barmode='stack', yaxis={'categoryorder':'array', 'categoryarray':top5_ordered}, font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

'''
A heatmap with the Type1 accorss generations
'''
# we create a pivot table counting the values of the frequency of each Type1
# by generation
cat_generation = pokemon_df[['Type1','Generation']].pivot_table(index='Type1', columns='Generation', aggfunc=len, fill_value=0)
cat_generation_fig = px.imshow(
    cat_generation,
    labels=dict(x='Generation', y='Type1', color='Frequency'),
    x=cat_generation.columns.to_list(),
    y=cat_generation.index.to_list(),
    text_auto=True
)
cat_generation_fig.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

'''
A Histogram showing the number of abilities by pokemon
'''
abilities_histogram = px.histogram(pokemon_df, x='N_Abilities', y='N_Abilities', color='Is_Legendary', 
                   barmode='group', histfunc='count', text_auto=True,
                   labels={'count': 'Count', 'N_Abilities': 'Number of Abilities', 'Is_Legendary': 'Is Legendary'})

abilities_histogram.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
'''
A query to get the damage_stats
'''

# query_damage_stats = '''
#     SELECT pk.pokedex_number, pk.name, pk.is_legendary, pkt.pokemon_type AS "Type1",
#     pkt2.pokemon_type AS "Type2", ds.*
#     FROM pokemon as pk
#     INNER JOIN pokemon_types as pkt ON pk.type1_id = pkt.id
#     INNER JOIN pokemon_types as pkt2 ON pk.type2_id = pkt2.id
#     INNER JOIN damage_stats as ds ON pk.pokedex_number = ds.pokedex_number
# '''
    
# pokemon_damage = pd.read_sql_query(sql=query_damage_stats, con=engine)

pokemon_damage = pd.read_csv('pokemon_damage.csv')
pokemon_damage.rename(str.title, axis='columns', inplace=True)

jumbotron_about = html.Div(
    dbc.Accordion(
        dbc.AccordionItem(
            dbc.Container([
                html.H1('Pokemon Dashboard'),
                html.P(
                    'This project consists of a data analysis exercise with the purpose of illustrating the findings contained in the data of about 800 pokemon through 7 generations.'
                ),
                html.Ul([
                    html.Li([
                        'Database',
                        html.Ul([
                            html.Li('A Python script that connects to a Postgresql database creates 4 tables and insert data from the data source'),
                            html.Li('A Python script to query the database and build the app dataframes'),
                            html.Li('Data source: https://www.kaggle.com/datasets/rounakbanik/pokemon')
                    ]),'Backend',
                        html.Ul([
                            html.Li('A SQLite instance to connect to the Postgresql Database and execute the queries for building the data structures'),
                            html.Li('A Padas DataFrames structure to store each pokemon stats'),
                            html.Li('A Plotly Dash instance to create a Dashboard'),
                            html.Li('The visualizations were ensambled with Python code for processing and transforming the data structures and passing it to Plotly objects which are rendered into the Dash object')
                        ])
                        ])
                ])
            ]),
            title='About'
        )
    )
    
)

def chart_info(info, collapsed, title) -> dbc.Accordion:
    accordion = dbc.Accordion([
        dbc.AccordionItem(
            html.P(
                info
            ), title=title
        )
    ], start_collapsed=collapsed)
    return accordion

def get_card(chart, info, collapsed, title) -> dbc.Card:
    card = dbc.Card([
        chart,
        dbc.CardBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    html.P(
                            info
                    ), title=title
                )
            ], start_collapsed=collapsed)
        ])
    ])
    return card

'''
Visualizations info
'''
bar_generations_accordion = 'Generation 6 has the highest number of pokemon and it seems that every odd generation has an increase of pokemon in comparison to the previous even generation'
capture_rate_atack_map = 'The tree map wraps each generation in one box, a second layer of boxes is contained wrapping the type and a third layer with the pokemon. The size of the box represents the capture rate of the pokemon and the color scale the Atack. the pokemon with the highest rate Floette of Fairy type has one of the lowest Attack, while Heracross of type Bug has one of the highest Attack and a decent capture rate'
type_stack_info = 'Normal + Flying is the most common type combination followed by Grass + Poison. We must keep in mind that 384 pokemon have only Type1'
type_combinations = 'Flying type pokemons are must common to have another primary type, followed by poison and ground'
top5_legendary = 'Psychic type are the most common legendary pokemon'
cat_generation = 'There are some types that do not exist in some generatios; steel and dark in gen 1, dragon and fairy in gens 2 and 3. As we saw before that flying pokemon are the most common type 2, there are only a few in gens 5 and 6  as type 1. Water '
capture_rate_footer = ''

app.layout = html.Div([
    html.Div([
        dbc.Row(dbc.Col(html.Img(src=app.get_asset_url('pokemon_logo.png')),width=12)),
        jumbotron_about,
        dbc.Row(
            [
                dbc.Col(get_card(dcc.Graph(id='bar-generations',figure=bar_gen), bar_generations_accordion, True, 'Number of Pokemon by Generation'),width=9),
                dbc.Col(html.Div([
                    html.Img(src=app.get_asset_url('charmander.png'),style={'width': '100%','object-fit':'cover'})
                ],style={'height':300, 'width':300}), width=3)
            ]
        ),
        get_card(dcc.Graph(id='map',figure=tree_chart), capture_rate_atack_map, True, 'Capture Rate - Attack Tree Map'),
        
        dbc.Row(
            [
                dbc.Col([get_card(dcc.Graph(id='types-stack',figure=types_bar), type_combinations, True, 'Must Common Type Combinations')], width=6),
                dbc.Col([get_card(dcc.Graph(id='top10-combinations', figure=top10_combinations), type_stack_info, True, 'Must Common Type Combinations')], width=6)
            ]
        ),
        dbc.Row(
            [
                dbc.Col([get_card(dcc.Graph(
                        id='top5-legendary',
                        figure=top5_leg_bar), top5_legendary, True, 'Frequency Legendary Pokemon')],width=5),
                dbc.Col([get_card(dcc.Graph(id='cat-generation',
                          figure=cat_generation_fig), cat_generation, True, 'Pokemon Frequency by Generation')], width=7)
            ]
        ),
        dbc.Card([
            dbc.CardHeader(dbc.RadioItems(
                id='x-axis',
                options=[{'label':'Legendary', 'value':1}, 
                         {'label':'no_Legendary', 'value': 0},
                         {'label':'All', 'value': 2}],
                value=2,
                inline=True)),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='boxplot_capture_rate'), width=6),
                    dbc.Col(dcc.Graph(id='boxplot_capture_rate_type'), width=6)]),
            ])
        ]),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(dcc.Graph(id='abilities_histogram', figure=abilities_histogram), width=7),
                    dbc.Col(dcc.Graph(id='weight_height'), width=5)        
                ])
            ])
        ]),
        dbc.Card([
            dbc.Row([
                dbc.Col(
                    [dbc.Row([
                        dbc.RadioItems(
                        id='bmi_radio',
                        options=[{'label': 'Top 10', 'value': 'Top 10'}, 
                                    {'label': 'Less 10', 'value': 'Less 10'}],
                        value='Top 10',
                        inline=True),
                    dbc.Row(dcc.Graph(id='BMI-histogram'))]
                    )]),
                dbc.Col([
                    dbc.Row(dbc.RadioItems(
                    id='stats_by_gen',
                    options=[{'label': 'Attack', 'value': 'Attack'}, 
                            {'label':'Defense', 'value': 'Defense'}, 
                            {'label':'HP', 'value': 'Hp'}, 
                            {'label':'Base Total', 'value': 'Base_Total'}],
                    value='Base_Total',
                    inline=True)),
                    dbc.Row(dcc.Graph(id='boxplot_stats'))])])
        ]),
        dbc.Card([
            dbc.CardBody([
                dbc.Row(dbc.RadioItems(
                    id='is_legendary_heat',
                    options=[
                        {'label': 'Legendary', 'value': 1},
                        {'label': 'No Legendary', 'value': 0 }
                    ],
                    inline=True,
                    value=0
                )),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='attributes_heatmap'), width=4),
                    dbc.Col(dcc.Graph(id='Type_attributes_heatmap'), width=8)
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='damage_heatmap'), width=4),
                    dbc.Col(dcc.Graph(id='top10_pokemon'), width=4),
                    dbc.Col(dcc.Graph(id='radar_chart'), width=4)
                ])
            ])
        ]),  
    ])
    ], className='dashboard_container')

@app.callback(
    Output('boxplot_capture_rate','figure'),
    Input('x-axis', 'value')
)
def boxplot_capture_rate(x):
    '''
    This callback generates a BoxPlot to illustrate the the capture rates by generation
    the Radioitems update the plot based on the user selection between all, legendary and not legendary
    '''
    if x == 2:
        df = pokemon_df
    else:
        df = pokemon_df[pokemon_df['Is_Legendary']==x]
    boxplot = px.box(df, x='Generation', y='Capture_Rate')
    boxplot.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return boxplot

@app.callback(
    Output('boxplot_capture_rate_type', 'figure'),
    Input('x-axis', 'value')
)
def boxplot_capure_rate_type(x):
    '''
    This callback generates a BoxPlot to illustrate the the capture rates by Type1
    the Radioitems update the plot based on the user selection between all, legendary and not legendary
    '''
    if x == 2:
        df = pokemon_df
    else:
        df = pokemon_df[pokemon_df['Is_Legendary']==x]
    boxplot = px.box(df, x='Type1', y='Capture_Rate')
    boxplot.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return boxplot

'''
Scatter plot with Wight and Height
'''
@app.callback(
    Output('weight_height', 'figure'),
    Input('x-axis', 'value')
)
def weight_height_scatter(legendary):
    if legendary == 2:
        df = pokemon_df
    else:
        df = pokemon_df[pokemon_df['Is_Legendary']==legendary]
        
    chart = px.scatter(df, x='Weight_Kg', y='Height_M')
    chart.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return chart

'''
BMI Hisogram
'''
@app.callback(
    Output('BMI-histogram', 'figure'),
    Input('bmi_radio', 'value')
)
def bmi_barchart(values):
    if values == 'Less 10':
        bmi = px.histogram(pokemon_df.sort_values(by='BMI', ascending=True)[:10].sort_values(by='BMI', ascending=False), x='BMI', y='Name', text_auto=True, orientation='h')
    else:
        bmi = px.histogram(pokemon_df.sort_values(by='BMI', ascending=False)[:10].sort_values(by='BMI', ascending=True), x='BMI', y='Name', text_auto=True, orientation='h')
    bmi.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return bmi   

'''
BoxPlot illustrating the stats by Generation updates on the type of stat chosen
'''     
@app.callback(
    Output('boxplot_stats', 'figure'),
    Input('stats_by_gen', 'value')
)
def boxplot_stats_by_gen(stats):
    boxplot = px.box(pokemon_df, x='Generation', y=stats, color='Is_Legendary')
    boxplot.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return boxplot

'''
Attibutes correlation heatmap
'''
@app.callback(
    Output('attributes_heatmap', 'figure'),
    Input('is_legendary_heat', 'value')
)
def attibutes_heatmap(value):
    attributes_corr = pokemon_df[pokemon_df['Is_Legendary']==value].loc[:,['Hp' , 'Speed', 'Defense', 'Attack', 'Sp_Defense', 'Sp_Attack']]
    
    corr = attributes_corr.corr()
    heat_fig = ff.create_annotated_heatmap(
        z=corr.to_numpy().round(2),
        x=list(corr.index.values),
        y=list(corr.columns.values),
        xgap=3, ygap=3,
        zmin=1, zmax=1,
        colorscale='earth',
        colorbar_thickness=30,
        colorbar_ticklen=3
    )
    heat_fig.update_layout(
        title_text = '<b>Correlation Matrix (Attributes)</b>',
        title_x=0.5,
        titlefont={'size':24},
        width=550, height=550,
        xaxis_showgrid=False,
        xaxis={'side':'bottom'},
        yaxis_showgrid=False,
        yaxis_autorange='reversed',
        paper_bgcolor=None
    )
    heat_fig.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return heat_fig

@app.callback(
    Output('Type_attributes_heatmap', 'figure'),
    Input('is_legendary_heat', 'value')
)
def heatmap_median_attributes(value):
    '''
    A heatmap with the Type1 accorss generations
    '''
    # we create a pivot table counting the values of the Median of each Type1
    # by Type1
    type_attribures = pokemon_df[pokemon_df['Is_Legendary']==value].loc[:,['Type1','Hp' , 'Speed', 'Defense', 'Attack', 'Sp_Defense', 'Sp_Attack']].pivot_table(index='Type1', aggfunc=np.median)
   
    type_attribures = px.imshow(
        type_attribures.transpose(),
        labels=dict(color='Value'),
        text_auto=True
    )
    type_attribures.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return type_attribures

@app.callback(
    Output('damage_heatmap', 'figure'),
    Input('is_legendary_heat', 'value')
)
def heatmap_pokemon_damage(value):
    damage_df = pokemon_damage[pokemon_damage['Is_Legendary']==value].loc[:,['Name', 'Type1', 'Against_Bug', 'Against_Dark', 'Against_Dragon', 'Against_Electric',
                                                                            'Against_Fairy', 'Against_Fight', 'Against_Fire', 'Against_Flying',
                                                                            'Against_Ghost', 'Against_Grass', 'Against_Ground', 'Against_Ice',
                                                                            'Against_Normal', 'Against_Poison', 'Against_Psychic', 'Against_Rock',
                                                                            'Against_Steel', 'Against_Water']].pivot_table(index='Type1', aggfunc=np.median)
    damage_heatmap = px.imshow(
        damage_df.transpose(),
        labels=dict(x='Type1', y='Damage', color='Value'),
        text_auto=True
    )
    damage_heatmap.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return damage_heatmap

@app.callback(
    Output('top10_pokemon', 'figure'),
    Input('is_legendary_heat', 'value')
)
def top_10_pokemon(value):
    top10_df = pokemon_df[pokemon_df['Is_Legendary'] == value].sort_values(by='Base_Total', ascending=False).iloc[:10][['Name', 'Base_Total']].sort_values(by='Base_Total', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top10_df['Base_Total'],
                y=top10_df['Name'],
                orientation='h',
                marker_color='rgb(55, 83, 109)'
                ))
    fig.update_layout(
        title='Top 10 Pokemon Base Total',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Pokemon Name',
            titlefont_size=16,
            tickfont_size=14
        ),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    fig.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

@app.callback(
    Output('radar_chart', 'figure'),
    Input('is_legendary_heat', 'value')
)
def radar_chart(value):
    categories = ['Attack', 'Defense', 'Speed', 'Sp_Attack', 'Sp_Defense', 'Hp']
    top10_df = pokemon_df[pokemon_df['Is_Legendary'] == value].sort_values(by='Base_Total', ascending=False).iloc[:10]

    radar_chart = go.Figure()
    iterations = 0
    name_index = 1

    for item in range(2):
        radar_chart.add_trace(go.Scatterpolar(
            r = top10_df.loc[:,categories].iloc[item,:].to_list(),
            theta=categories,
            fill='toself',
            name=top10_df.iloc[item,:].Name
        ))

    radar_chart.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 200]
            )
        )
    )
    radar_chart.update_layout(font=chart_layout_font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return radar_chart
    



if __name__ == '__main__':
    app.run_server(debug=False)