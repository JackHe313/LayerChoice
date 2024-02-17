# Calculate CT score

Specify Only the Data Source
```
python ct.py data_source_name
```
This requires data_source_name to be in the default source list

Specify Data Source and Custom Path (recommanded)
```
python script_name.py data_source_name --custom_path /path/to/data
```
This command specifies a custom path for the data, overriding the default path.

Specify Data Source and Enable CT score saving
```
python script_name.py data_source --save
````
This command will save the model name and CT-score to a file, in addition to processing the data source.




