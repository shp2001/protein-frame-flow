import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description='Merging two csv files')

parser.add_argument(
    '--csv1',
    type=str)
parser.add_argument(
    '--csv2',
    type=str)
parser.add_argument(
    '--write_file',
    type=str)

# valid csv file도 train csv file의 열과 맞추기 위해 
# parser.add_argument(
#     '--valid_csv',
#     type=str
# )
# parser.add_argument(
#     '--write_valid_file',
#     type=str)

def main(args):
    csv1 = pd.read_csv(args.csv1)
    csv2 = pd.read_csv(args.csv2)

    # 두 DataFrame의 열을 동일하게 맞추기 (없는 열에는 None 채우기)
    merged = pd.concat([csv1, csv2], ignore_index=True)

    # 결과를 새로운 CSV 파일로 저장
    merged.to_csv(args.write_file, index=False, na_rep="None")

    # valiation csv에 없는 열 추가
    # valid_csv = pd.read_csv(args.valid_csv)
    # missing_columns = set(merged.columns) - set(valid_csv.columns)

    # for col in missing_columns:
    #     valid_csv[col] = None
    
    # valid_csv.to_csv(args.write_valid_file, index=False, na_rep="None")
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)