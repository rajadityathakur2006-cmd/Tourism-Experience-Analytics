"""
Tourism Experience Analytics - Data Cleaning & Consolidation
Fixes discovered during inspection:
 1. Item.AttractionCityId does NOT map into City.xlsx's id space (that table's
    ids 1/2/3 belong to Cameroon/Chad). All 30 attractions are actually in
    Indonesia (Bali / Malang / Yogyakarta), inferred from AttractionAddress.
    -> Replaced with a manual, address-verified region/country mapping.
 2. User.CityId has 4 nulls -> filled with 0 ("unknown" placeholder, consistent
    with how City/Country/Region/Continent/Mode already encode unknowns as id 0).
 3. User.RegionId has 19 rows = 0 ("unknown") -> left as-is, labelled via the
    existing '-' placeholder row in Region.xlsx rather than dropped (dropping
    would lose 19 otherwise-valid users).
 4. Updated_Item.xlsx (1,698 rows, dirty AttractionTypeId mixing ints and
    strings) is NOT used - it has zero overlap with real transactions and
    fixing it would be wasted effort for this project's scope.
 5. Transaction, City-name duplicates, and referential keys elsewhere were
    already consistent - no further action needed there.
"""
import pandas as pd

BASE = 'data/raw/'
OUT = 'data/cleaned/'

txn = pd.read_excel(BASE+'Transaction.xlsx')
user = pd.read_excel(BASE+'User.xlsx')
city = pd.read_excel(BASE+'City.xlsx')
country = pd.read_excel(BASE+'Country.xlsx')
region = pd.read_excel(BASE+'Region.xlsx')
continent = pd.read_excel(BASE+'Continent.xlsx')
item = pd.read_excel(BASE+'Item.xlsx')
typ = pd.read_excel(BASE+'Type.xlsx')
mode = pd.read_excel(BASE+'Mode.xlsx')

# --- Users / geography ---
user['CityId'] = user['CityId'].fillna(0).astype(int)
city['CityName'] = city['CityName'].astype(str).str.strip()

geo = (user
    .merge(continent, on='ContinentId', how='left')
    .merge(region[['RegionId', 'Region']], on='RegionId', how='left')
    .merge(country[['CountryId', 'Country']], on='CountryId', how='left')
    .merge(city[['CityId', 'CityName']], on='CityId', how='left')
    .rename(columns={'CityName': 'UserCity', 'Country': 'UserCountry',
                      'Region': 'UserRegion', 'Continent': 'UserContinent'}))
geo['UserRegion'] = geo['UserRegion'].fillna('Unknown')
geo['UserCity'] = geo['UserCity'].replace('-', 'Unknown').fillna('Unknown')

# --- Attractions (manual fix for broken AttractionCityId) ---
CITY_FIX = {1: 'Bali', 2: 'Malang (East Java)', 3: 'Yogyakarta'}
item['AttractionRegion'] = item['AttractionCityId'].map(CITY_FIX)
item['AttractionCountry'] = 'Indonesia'

attr = item.merge(typ, on='AttractionTypeId', how='left')
attr = attr[['AttractionId', 'Attraction', 'AttractionType', 'AttractionRegion',
             'AttractionCountry', 'AttractionAddress']]

# --- Visit mode labels ---
mode_map = mode.set_index('VisitModeId')['VisitMode'].to_dict()
txn['VisitModeLabel'] = txn['VisitMode'].map(mode_map)

# --- Master (transaction-level) table ---
master = (txn
    .merge(geo[['UserId', 'UserContinent', 'UserRegion', 'UserCountry', 'UserCity']],
           on='UserId', how='left')
    .merge(attr, on='AttractionId', how='left'))

assert master.isnull().sum().sum() == 0, "Unexpected nulls after cleaning"

geo.to_csv(OUT+'users_clean.csv', index=False)
attr.to_csv(OUT+'attractions_clean.csv', index=False)
master.to_csv(OUT+'master_clean.csv', index=False)

print("users_clean:", geo.shape)
print("attractions_clean:", attr.shape)
print("master_clean:", master.shape)
print("\nAttraction region distribution:\n", attr.AttractionRegion.value_counts())
