## Objective
- Query anything from database.

## Function

### 1. SQL generated by LLM
- Try with few examples and find one that best fits for your case.
- Tweak name and description to find better result of query

#### simple (with no validation)
```yaml
- spec:
    name: query_histories_from_db
    description: >-
      Use this function to query histories from Home Assistant SQLite database.
      Example:
        Question: How long was livingroom light on in Nov 15?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated, s.state, old.state as prev_state FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'switch.livingroom' AND s.state != old.state AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') BETWEEN '2023-11-15 00:00:00' AND '2023-11-15 23:59:59'
    parameters:
      type: object
      properties:
        query:
          type: string
          description: A fully formed SQL query.
  function:
    type: sqlite
```

```yaml
- spec:
    name: query_histories_from_db
    description: >-
      Use this function to query histories from Home Assistant SQLite database.
      Example:
        Question: When did bedroom light turn on?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated_ts FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'light.bedroom' AND s.state = 'on' AND s.state != old.state ORDER BY s.last_updated_ts DESC LIMIT 1
        Question: Was livingroom light on at 9 am?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated, s.state FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'switch.livingroom' AND s.state != old.state AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') < '2023-11-17 08:00:00' ORDER BY s.last_updated_ts DESC LIMIT 1
    parameters:
      type: object
      properties:
        query:
          type: string
          description: A fully formed SQL query.
  function:
    type: sqlite
```

#### with minimum validation (still not enough)
```yaml
- spec:
    name: query_histories_from_db
    description: >-
      Use this function to query histories from Home Assistant SQLite database.
      Example:
        Question: How long was livingroom light on in Nov 15?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated, s.state, old.state as prev_state FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'switch.livingroom' AND s.state != old.state AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') BETWEEN '2023-11-15 00:00:00' AND '2023-11-15 23:59:59'
    parameters:
      type: object
      properties:
        query:
          type: string
          description: A fully formed SQL query.
  function:
    type: sqlite
    query: >-
      {%- if is_exposed_entity_in_query(query) -%}
        {{ query }}
      {%- else -%}
        {{ raise("entity_id should be exposed.") }}
      {%- endif -%}
```

```yaml
- spec:
    name: query_histories_from_db
    description: >-
      Use this function to query histories from Home Assistant SQLite database.
      Example:
        Question: When did bedroom light turn on?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated_ts FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'light.bedroom' AND s.state = 'on' AND s.state != old.state ORDER BY s.last_updated_ts DESC LIMIT 1
        Question: Was livingroom light on at 9 am?
        Answer: SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') last_updated, s.state FROM states s INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id INNER JOIN states old ON s.old_state_id = old.state_id WHERE sm.entity_id = 'switch.livingroom' AND s.state != old.state AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') < '2023-11-17 08:00:00' ORDER BY s.last_updated_ts DESC LIMIT 1
    parameters:
      type: object
      properties:
        query:
          type: string
          description: A fully formed SQL query.
  function:
    type: sqlite
    query: >-
      {%- if is_exposed_entity_in_query(query) -%}
        {{ query }}
      {%- else -%}
        {{ raise("entity_id should be exposed.") }}
      {%- endif -%}
```


### 2. Defined SQL manually

#### 2-1. get_state_at_time
<img width="300" src="https://github.com/mLupine/OmniConv/assets/2917984/19fac845-5cee-4d84-98b5-1e18994bb2ee">
<img width="300" src="https://github.com/mLupine/OmniConv/assets/2917984/af8a26d1-0525-4411-b323-29be92d8f368">

```yaml
- spec:
    name: get_state_at_time
    description: >
      Use this function to get state at time
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: The target entity
        datetime:
          type: string
          description: The datetime in '%Y-%m-%d %H:%M:%S' format
      required:
        - entity_id
        - datetime
        - limit
  function:
    type: sqlite
    query: >-
      {%- if is_exposed(entity_id) -%}
        SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') as state_updated_at, s.state
        FROM states s
                 INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                 INNER JOIN states old ON s.old_state_id = old.state_id
        WHERE sm.entity_id = '{{entity_id}}'
          AND s.state != old.state
          AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') < '{{datetime}}'
        ORDER BY s.last_updated_ts DESC
        LIMIT 1
      {%- else -%}
        {{ raise("entity_id should be exposed.") }}
      {%- endif -%}
```


#### 2-2. get_states_between
<img width="300" src="https://github.com/mLupine/OmniConv/assets/2917984/d504b372-460b-45b8-8705-027548bc0c52">

```yaml
- spec:
    name: get_states_between
    description: >
      Use this function to get non-numeric states between two dates.
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: The target entity
        state:
          type: string
          description: The state
        state_operator:
          type: string
          description: The state operator
          enum:
          - ">"
          - "<"
          - "="
          - ">="
          - "<="
        start_datetime:
          type: string
          description: The start datetime in '%Y-%m-%d %H:%M:%S' format
        end_datetime:
          type: string
          description: The end datetime in '%Y-%m-%d %H:%M:%S' format
        order:
          type: string
          description: The order of datetime, defaults to desc
          enum:
          - asc
          - desc
        page:
          type: integer
          description: The page number
        limit:
          type: integer
          description: The page size defaults to 10
      required:
        - entity_id
        - start_datetime
        - end_datetime
        - order
        - page
        - limit
  function:
    type: composite
    sequence:
      - type: sqlite
        query: >-
          {%- if is_exposed(entity_id) -%}
            SELECT datetime(s.last_updated_ts, 'unixepoch', 'localtime') as updated_at, s.state
            FROM states s
                     INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                     INNER JOIN states old ON s.old_state_id = old.state_id
            WHERE sm.entity_id = '{{entity_id}}'
              AND s.state != old.state
              AND (('{{state | default('')}}' = '') OR (s.state {{state_operator | default('=')}} '{{state | default('')}}'))
              AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') >= '{{start_datetime}}'
              AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') < '{{end_datetime}}'
            ORDER BY s.last_updated_ts {{order}}
            LIMIT {{(page-1) * limit}}, {{limit}}
          {%- else -%}
            {{ raise("entity_id should be exposed.") }}
          {%- endif -%}
        response_variable: data
      - type: sqlite
        single: true
        query: >-
          SELECT count(*) as count
          FROM states s
                   INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                   INNER JOIN states old ON s.old_state_id = old.state_id
          WHERE sm.entity_id = '{{entity_id}}'
            AND s.state != old.state
            AND (('{{state | default('')}}' = '') OR (s.state {{state_operator | default('=')}} '{{state | default('')}}'))
            AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') >= '{{start_datetime}}'
            AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') < '{{end_datetime}}'
        response_variable: total
      - type: template
        value_template: '{"data": {{data}}, "total": {{total.count}}}'
```

#### 2-3. get_total_time_of_entity_state
<img width="300" src="https://github.com/mLupine/OmniConv/assets/2917984/42eab8bc-3326-4d70-b065-a4788060a49b">
<img width="300" src="https://github.com/mLupine/OmniConv/assets/2917984/1e01fce1-7891-4d87-bfea-cd52c5c73592">

```yaml
- spec:
    name: get_total_time_of_entity_state
    description: >
      Use this function to get total time of state of entity between two dates
    parameters:
      type: object
      properties:
        entity_id:
          type: string
          description: The target entity
        state:
          type: string
          description: The non-numeric target state
        start_datetime:
          type: string
          description: The start datetime in '%Y-%m-%d %H:%M:%S' format
        end_datetime:
          type: string
          description: The end datetime in '%Y-%m-%d %H:%M:%S' format
      required:
        - entity_id
        - state
        - start_datetime
        - end_datetime
  function:
    type: composite
    sequence:
      - type: sqlite
        query: >-
          {%- if is_exposed(entity_id) -%}
            WITH stat_data AS (
              WITH lead_data AS (
                SELECT datetime(old.last_updated_ts, 'unixepoch', 'localtime') AS prev_last_updated,
                  old.state AS prev_state,
                  datetime(s.last_updated_ts, 'unixepoch', 'localtime') AS last_updated,
                  s.state,
                  COALESCE(LEAD(datetime(s.last_updated_ts, 'unixepoch', 'localtime')) OVER (ORDER BY s.last_updated), '{{end_datetime}}') AS lead_last_updated,
                  LEAD(s.state) OVER (ORDER BY s.last_updated) AS lead_state
                FROM states s
                  INNER JOIN states_meta sm ON s.metadata_id = sm.metadata_id
                  INNER JOIN states old ON s.old_state_id = old.state_id
                WHERE sm.entity_id = '{{entity_id}}'
                  AND s.state != old.state
                  AND datetime(s.last_updated_ts, 'unixepoch', 'localtime') BETWEEN '{{start_datetime}}' AND '{{end_datetime}}'
                )
              SELECT max(prev_last_updated, '{{start_datetime}}') AS prev_last_updated,
                prev_state,
                last_updated AS last_updated,
                state
              FROM lead_data
              WHERE last_updated = (SELECT MIN(last_updated) FROM lead_data)

              UNION ALL

              SELECT last_updated AS prev_last_updated, state AS prev_state, min(lead_last_updated, strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')) AS last_updated, lead_state AS state
              FROM lead_data
            )
            SELECT SUM(CASE WHEN prev_state = '{{state}}' THEN cast(strftime('%s', last_updated, 'utc') as real) - cast(strftime('%s', prev_last_updated, 'utc') as real) ELSE 0 END) AS total_time_in_sec FROM stat_data
          {%- else -%}
            {{ raise("entity_id should be exposed.") }}
          {%- endif -%}
        response_variable: result
      - type: template
        value_template: >-
          {%- if result and result[0] and result[0].total_time_in_sec -%}
            {%- set duration = result[0].total_time_in_sec | int -%}

            {%- set days = (duration // 86400) | int -%}
            {%- set hours = ((duration % 86400) // 3600) | int -%}
            {%- set minutes = ((duration % 3600) // 60) | int -%}
            {%- set remaining_seconds = (duration % 60) | int -%}

            {{ "{0}d ".format(days) if days > 0 else "" }}{{ "{0}h ".format(hours) if hours > 0 else "" }}{{ "{0}m ".format(minutes) if minutes > 0 else "" }}{{ "{0}s".format(remaining_seconds) if remaining_seconds > 0 else "" }}
          {%- else -%}
            unkown
          {%- endif -%}
```
