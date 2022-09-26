import psycopg2
import pandas as pd

hostname = 'pokemon.cny93cbmckvv.ca-central-1.rds.amazonaws.com'
database = 'pokemon'
username = 'postgres'
pwd = 'Elichu$123'
port_id = 5432
conn = None
cur = None

classification = pd.read_excel('Book1.xlsx','clasification')
types = pd.read_excel('Book1.xlsx', 'type')

try:
    conn = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=pwd,
        port=port_id
    )
    
    cur = conn.cursor()
    
    # CLASSIFICATION TABLE
    # if exist we delete it    
    cur.execute('DROP TABLE IF EXISTS classification')
    
    create_classification = ''' CREATE TABLE IF NOT EXISTS classification (
        id                  serial PRIMARY KEY,
        classification      VARCHAR(70) UNIQUE NOT NULL
    )
    '''
    cur.execute(create_classification)
    
    # TYPES TABLE
    # if exist we delete it
    cur.execute('DROP TABLE IF EXISTS pokemon_types')
    
    create_types = ''' CREATE TABLE IF NOT EXISTS pokemon_types (
        id                  INTEGER PRIMARY KEY,
        pokemon_type        VARCHAR(50) UNIQUE
    )
    '''
    
    cur.execute(create_types)
    
    cur.execute('DROP TABLE IF EXISTS pokemon CASCADE')
    
    create_pokemon = ''' CREATE TABLE IF NOT EXISTS pokemon (
        pokedex_number              SERIAL PRIMARY KEY,
        name                        VARCHAR(50) UNIQUE NOT NULL NOT NULL,
        classification_id           INTEGER REFERENCES classification(id) ,
        type1_id                    INTEGER REFERENCES pokemon_types(id) NOT NULL,
        type2_id                    INTEGER,
        attack                      REAL NOT NULL,
        defense                     REAL NOT NULL,
        experience_growth           REAL NOT NULL,
        base_egg                    REAL NOT NULL,
        base_happiness              REAL NOT NULL,
        base_total                  REAL NOT NULL,
        capture_rate                REAL,
        height_m                    REAL,
        weight_kg                   REAL,
        generation                  INTEGER NOT NULL,
        is_legendary                INTEGER NOT NULL,
        hp                          REAL NOT NULL,
        percentage_male             REAL,
        sp_attack                   REAL NOT NULL,
        sp_defense                  REAL NOT NULL,
        speed                       REAL NOT NULL,
        abilities                   VARCHAR(200)
    )
    '''
    
    cur.execute(create_pokemon)
    
    cur.execute('DROP TABLE IF EXISTS damage_stats')
    
    create_damage = ''' CREATE TABLE IF NOT EXISTS damage_stats (
        pokedex_number              INTEGER PRIMARY KEY REFERENCES pokemon(pokedex_number),
        against_bug                 REAL,
        against_dark                REAL,
        against_dragon              REAL,
        against_electric            REAL,
        against_fairy               REAL,
        against_fight               REAL,
        against_fire                REAL,
        against_flying              REAL,
        against_ghost               REAL,
        against_grass               REAL,
        against_ground              REAL,
        against_ice                 REAL,
        against_normal              REAL,
        against_poison              REAL,
        against_psychic             REAL,
        against_rock                REAL,
        against_steel               REAL,
        against_water               REAL
    )
    '''
    
    cur.execute(create_damage)
    
    # INSERT data to classification
    '''
    we create a list of tuples with the classification data
    '''
    classification_list = []
    for index, row in classification.iterrows():
        classification_list.append((row['classfication'],))
    
    # We pass a tuple for each iteration to the execute method
    insert_classification = 'INSERT INTO classification (classification) VALUES(%s)'
    for item in classification_list:
        cur.execute(insert_classification, item)
        
    types_list = []
    for index, row in types.iterrows():
        types_list.append((row['type_id'], row['type1']))
        
    insert_types = 'INSERT INTO pokemon_types (id, pokemon_type) VALUES(%s, %s)'
    for item in types_list:
        cur.execute(insert_types, item)
        
    # Load the pokemon table in a dataframe
    pokemon_df = pd.read_excel('Book1.xlsx', 'pokemon_')
    pokemon_df.drop(['japanese_name'], axis=1, inplace=True)
    pokemon_df.drop(['pokedex_number'], axis=1, inplace=True)
    # create a list of tuples with each row values
    _tuple = [tuple(x) for x in pokemon_df.values]
    # create the SQL script INSERT
    insert_pokemon = '''INSERT INTO pokemon (name, classification_id, type1_id, type2_id, attack, defense, 
                                    experience_growth, base_egg, base_happiness, base_total, capture_rate, 
                                    height_m, weight_kg, generation, is_legendary, hp, percentage_male, 
                                    sp_attack, sp_defense, speed, abilities) 
                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''

    # Loop through the list of tuples and execute the script 
    for item in _tuple:
        cur.execute(insert_pokemon, item)
        
    # Load the damage_stats table in a dataframe
    damage_df = pd.read_excel('Book1.xlsx', 'damage_stats')
    # create a list of tuples with each row values
    _tuple = [tuple(x) for x in damage_df.values]
    # create the SQL script INSERT
    
    insert_damage = ''' INSERT INTO damage_stats (pokedex_number,
                        against_bug, against_dark, against_dragon, against_electric, against_fairy, against_fight, against_fire,
                        against_flying, against_ghost, against_grass, against_ground, against_ice, against_normal, against_poison,
                        against_psychic, against_rock, against_steel, against_water) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    '''
    
    for item in _tuple:
        cur.execute(insert_damage, item)
    
    conn.commit()

except Exception as error:
    print(error)
    print('Ups')
finally:
    if cur is not None:
        cur.close()
    if conn is not None:
        conn.close()
    
