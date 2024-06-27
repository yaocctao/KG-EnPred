import clickhouse_connect
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import Tuple, Dict
from collections import namedtuple
import os



def main(save_path: str, relations: str, train_radio: str, valid_radio: str, test_radio: str, intervals: int,  start_time_str: str = '2021-04-30 00:00:00' , end_time_str: str = '2021-06-06 00:00:00'):
    client = clickhouse_connect.get_client(
        host='10.10.120.29',username='admin', password='admin123..',database='station_data',send_receive_timeout="1000000")

    sql = f"""
        SELECT *
        FROM station_data.fix_etc_in_May feim
        WHERE obuplate GLOBAL IN (
            SELECT
            obuplate 
            FROM 
                station_data.fix_etc_in_May 
            WHERE
                obuplate GLOBAL NOT IN (
                    SELECT 
                        obuplate
                    FROM 
                        station_data.fix_etc_in_May
                    WHERE
                        passid LIKE '010000FFFFFFFFFFFFFFFF%'
                        OR ennetid != '3501'
                        OR enstation = '0000'
                        OR stationname = '0'
                        OR obuplate = '0'
                        OR obuplate LIKE '未确定%'
                    GROUP BY 
                        obuplate 
                )
            GROUP BY 
                obuplate 
            HAVING 
                COUNT(*) > 42
        )
        AND entime >= '{start_time_str}'
        AND entime <= '{end_time_str}'
        ORDER BY 
            obuplate, entime
        """
    
    enity2idFilePath = os.path.join(save_path, "enity2id.txt")
    enity2idFile = open(enity2idFilePath, 'w+')
    relation2idFilePath = os.path.join(save_path, "relation2id.txt")
    relation2idFile = open(relation2idFilePath, 'w+')
    time2idFilePath = os.path.join(save_path, "time2id.txt")
    time2idFile = open(time2idFilePath, 'w+')

    trainFilePath = os.path.join(save_path, "train")
    trainFile = open(trainFilePath, 'w+')
    validFilePath = os.path.join(save_path, "valid")
    validFile = open(validFilePath, 'w+')
    testFilePath = os.path.join(save_path, "test")
    testFile = open(testFilePath, 'w+')

    df_stream = client.query_df_stream(sql)
    column_names = df_stream.source.column_names
    print(column_names)

    enity2id = dict()
    relation2id = dict()
    for r in relations:
        relation2id[r] = len(relation2id)
    CarInfo = namedtuple('Carinfo', ['name','data'])
    
    global enityCount
    enityCount = 0
    last_car_info = None
    
    def get_enity_id(enity_name: str) -> int:
        if enity_name in enity2id:
            return enity2id[enity_name]
        else:
            enity2id[enity_name] = globals()['enityCount']
            enity_id = globals()['enityCount']
            globals()['enityCount'] += 1
        return enity_id
    
    def writeUndirect2File(last_car_info):
        cat_data_lenght = len(last_car_info.data)
        #write to train
        for i in range(0, int(cat_data_lenght*train_radio)):
            trainFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
            trainFile.write(f"{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['relation'] + len(relation2id)}\t{last_car_info.data[i]['head']}\t{last_car_info.data[i]['time']}\n")
        #write to valid
        for i in range(int(cat_data_lenght*train_radio), int(cat_data_lenght*(train_radio+valid_radio))):
            testFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
            testFile.write(f"{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['relation'] + len(relation2id)}\t{last_car_info.data[i]['head']}\t{last_car_info.data[i]['time']}\n")
        #write to test
        for i in range(int(cat_data_lenght*(train_radio+valid_radio)), cat_data_lenght):
            validFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
            validFile.write(f"{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['relation'] + len(relation2id)}\t{last_car_info.data[i]['head']}\t{last_car_info.data[i]['time']}\n")
    
    def write2file(last_car_info):
        cat_data_lenght = len(last_car_info.data)
        #write to train
        for i in range(0, int(cat_data_lenght*train_radio)):
            trainFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
        #write to valid
        for i in range(int(cat_data_lenght*train_radio), int(cat_data_lenght*(train_radio+valid_radio))):
            testFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
        #write to test
        for i in range(int(cat_data_lenght*(train_radio+valid_radio)), cat_data_lenght):
            validFile.write(f"{last_car_info.data[i]['head']}\t{last_car_info.data[i]['relation']}\t{last_car_info.data[i]['tail']}\t{last_car_info.data[i]['time']}\n")
    
    id2intervals, intervals2id = generate_time_intervals(start_time_str, end_time_str, intervals)
    
    with df_stream:
        for _,df in tqdm(enumerate(df_stream)):
            for _, row in df.iterrows():
                carId = get_enity_id(row['obuplate'])
                enStationId = get_enity_id(row['enstation'].decode('utf-8'))
                if last_car_info == None:
                    # last_car_info = CarInfo(name=row['obuplate'], data=[{"head": str(enStationId), "relation": relation2id['is_enter'], "tail": str(carId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)}])
                    last_car_info = CarInfo(name=row['obuplate'], data=[{"head": str(carId), "relation": relation2id['is_enter'], "tail": str(enStationId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)}])
                elif last_car_info.name == row['obuplate']:
                    # last_car_info.data.append({"head": str(enStationId), "relation": relation2id['is_enter'], "tail": str(carId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)})
                    last_car_info.data.append({"head": str(carId), "relation": relation2id['is_enter'], "tail": str(enStationId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)})
                else:
                    write2file(last_car_info)
                    # last_car_info = CarInfo(name=row['obuplate'], data=[{"head": str(enStationId), "relation": relation2id['is_enter'], "tail": str(carId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)}])
                    last_car_info = CarInfo(name=row['obuplate'], data=[{"head": str(carId), "relation": relation2id['is_enter'], "tail": str(enStationId), "time": get_timestamp_id(intervals2id, row['entime'], intervals)}])

        write2file(last_car_info)
        trainFile.close()
        validFile.close()
        testFile.close()
        print(f"Total enity count: {enityCount}")
        enity2idFile.writelines(list(map(lambda x: f"{x[0]}\t{x[1]}\n", enity2id.items())))
        relation2idFile.writelines(list(map(lambda x: f"{x[0]}\t{x[1]}\n", relation2id.items())))
        time2idFile.writelines(list(map(lambda x: f"{str(x[0])}\t{x[1]}\n", id2intervals.items())))

def generate_time_intervals(start_time_str: str, end_time_str: str, intervals: str) -> Tuple[Dict, Dict]:
    
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    id2intervals = {}
    intervals2id = {}
    current_time = start_time
    interval_id = 0
    
    while current_time < end_time:
        interval_start = current_time
        interval_end = current_time + timedelta(hours=intervals)
        if interval_end > end_time:
            interval_end = end_time
        
        interval_key = interval_id
        id2intervals[interval_key] = (interval_start.strftime('%Y-%m-%d %H:%M:%S'), interval_end.strftime('%Y-%m-%d %H:%M:%S'))
        intervals2id[(interval_start.strftime('%Y-%m-%d %H:%M:%S'), interval_end.strftime('%Y-%m-%d %H:%M:%S'))] = interval_key

        current_time += timedelta(hours=intervals)
        interval_id += 1

    return (id2intervals, intervals2id)
        
def get_timestamp_id(intervals2id: Dict[Tuple[str, str], int], datetime: datetime, intervals: int) -> int:
    start_datetime = datetime.replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    end_datetime = (datetime+timedelta(hours=intervals)).replace(minute=0, second=0, microsecond=0).strftime('%Y-%m-%d %H:%M:%S')
    search_time = (start_datetime, end_datetime)
    
    if search_time not in intervals2id:
        print("Error: Time interval not found.")
        return -1
    
    return str(intervals2id[search_time])

if __name__ == "__main__":
    
    save_path = "src_data/fujian"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    relations = ['is_enter']

    train_radio = 0.7
    valid_radio = 0.15
    test_radio = 0.15
    start_time_str = '2021-05-01 00:00:00'
    end_time_str = '2021-06-06 00:00:00'
    intervals = 1

    main(save_path, relations, train_radio, valid_radio, test_radio, intervals, start_time_str, end_time_str)