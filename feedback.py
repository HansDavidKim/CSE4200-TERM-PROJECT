import pandas as pd
import numpy as np
import io

class FeedbackProcessor:
    """
    추천 시스템 데이터셋에서 Explicit 및 Implicit Feedback을 추출하고,
    User-Item Matrix를 생성하는 클래스.
    """
    def __init__(self, file_path):
        """
        초기화 메서드. 데이터 파일 경로를 설정하고 데이터를 로드합니다.

        :param file_path: 로드할 CSV 파일의 경로 (예: 'data_200_200_30.csv')
        """
        self.file_path = file_path
        self.df = None
        self.explicit_df = None
        self.implicit_df = None
        self.explicit_matrix = None
        self.implicit_matrix = None

    def load_data(self):
        """
        지정된 경로에서 CSV 파일을 로드합니다.
        """
        print(f"데이터 로드 시작: {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path)
            print("데이터 로드 성공.")
        except FileNotFoundError:
            print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {self.file_path}")
            self.df = None
            return False
        except Exception as e:
            print(f"Error: 데이터 로드 중 오류 발생: {e}")
            self.df = None
            return False
        return True

    def _prepare_data(self):
        """
        'action' 컬럼의 문자열 값을 정수 리스트로 변환하여 'action_list' 컬럼을 생성합니다.
        """
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. load_data()를 먼저 실행해주세요.")

        print("데이터 준비 (action 리스트 변환) 중...")
        # 쉼표로 구분된 문자열을 파싱하고 정수로 변환
        self.df['action_list'] = self.df['action'].apply(
            lambda x: [int(i) for i in str(x).split(',')]
        )
        print("데이터 준비 완료.")

    def extract_feedback(self):
        """
        로드된 데이터로부터 Explicit Feedback과 Implicit Feedback을 추출합니다.
        """
        self._prepare_data()

        explicit_feedback_list = []
        implicit_feedback_list = []

        print("피드백 추출 시작...")

        # 데이터프레임을 행 단위로 순회
        for index, row in self.df.iterrows():
            user_id = row['user_id']
            step = row['step']
            reward = row['reward']
            actions = row['action_list']
            num_actions = len(actions)

            for i, item_id in enumerate(actions):
                # i는 0부터 num_actions - 1까지의 인덱스로, resp_i 컬럼셋에 대응
                prefix = f'resp_{i}_'

                # 'resp_i_click_doc_id' 값이 존재하는지 확인
                responsed = row.get(prefix + 'click_doc_id', -1) # -1을 기본값으로 사용

                # --- 1. Explicit Feedback 추출 (Reward 기반) ---
                # 'click_doc_id'가 -1이 아니면 (응답했다면) 해당 행의 reward 사용
                # 그렇지 않으면 reward 0 사용
                current_reward = reward if responsed != -1 else 0
                explicit_feedback_list.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'step': step,
                    'reward': current_reward
                })

                # --- 2. Implicit Feedback 추출 (Interaction 기반) ---
                # 응답 컬럼 확인 및 값 가져오기
                clicked = row.get(prefix + 'click', 0)
                watched = row.get(prefix + 'watch', 0)
                liked = row.get(prefix + 'liked', 0)

                # response 판단: click_doc_id != -1 이고, (click=1 OR watch!=0 OR liked!=0) 이면 1, 아니면 0
                response = 1 if responsed != -1 and (clicked == 1 or watched != 0 or liked != 0) else 0

                implicit_feedback_list.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'step': step,
                    'response': response
                })

        # 결과 데이터프레임 생성
        self.explicit_df = pd.DataFrame(explicit_feedback_list)
        self.implicit_df = pd.DataFrame(implicit_feedback_list)

        print("피드백 추출 완료.")
        print(f"총 Explicit Feedback 수: {len(self.explicit_df)}")
        print(f"총 Implicit Feedback 수: {len(self.implicit_df)}")

    def create_user_item_matrix(self):
        """
        추출된 피드백 데이터프레임을 사용하여 User-Item Matrix를 생성합니다.
        """
        if self.explicit_df is None or self.implicit_df is None:
            raise ValueError("피드백 데이터가 추출되지 않았습니다. extract_feedback()을 먼저 실행해주세요.")

        print("User-Item Matrix 생성 시작...")

        # --- Explicit User-Item Matrix (Reward) ---
        # aggfunc='max'를 사용하여 동일 user/item/step 쌍이 여러 개 있을 경우 (이 코드 구조에서는 발생하지 않지만 안전을 위해) 최대 reward를 사용
        self.explicit_matrix = self.explicit_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='reward',
            aggfunc='max',
            fill_value=0
        )

        # --- Implicit User-Item Matrix (Response) ---
        # response가 0 또는 1이므로 'max'를 사용하여 상호작용 여부(1 또는 0)를 결정
        self.implicit_matrix = self.implicit_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='response',
            aggfunc='max',
            fill_value=0
        )

        print("User-Item Matrix 생성 완료.")
        print(f"Explicit Matrix 크기: {self.explicit_matrix.shape}")
        print(f"Implicit Matrix 크기: {self.implicit_matrix.shape}")

    def get_matrices(self):
        """
        생성된 Explicit 및 Implicit User-Item Matrix를 반환합니다.

        :return: (explicit_matrix, implicit_matrix) 튜플
        """
        if self.explicit_matrix is None or self.implicit_matrix is None:
            raise ValueError("User-Item Matrix가 생성되지 않았습니다. create_user_item_matrix()를 먼저 실행해주세요.")
        return self.explicit_matrix, self.implicit_matrix

    def get_feedback_dfs(self):
        """
        추출된 Explicit 및 Implicit Feedback 데이터프레임을 반환합니다.

        :return: (explicit_df, implicit_df) 튜플
        """
        if self.explicit_df is None or self.implicit_df is None:
            raise ValueError("피드백 데이터가 추출되지 않았습니다. extract_feedback()을 먼저 실행해주세요.")
        return self.explicit_df, self.implicit_df

    
# if __name__ == '__main__':
#     file_name = 'data_200_200_30.csv'
#     processor = FeedbackProcessor(file_name)

#     # 1. 데이터 로드
#     if processor.load_data():
#         # 2. 피드백 추출
#         processor.extract_feedback()

#         print("\n" + "="*50)
#         print("Explicit Feedback (상위 10개):")
#         print(processor.explicit_df.head(10))
#         print("\n" + "="*50)
#         print("Implicit Feedback (상위 10개):")
#         print(processor.implicit_df.head(10))
#         print("="*50)

#         # 3. User-Item Matrix 생성
#         processor.create_user_item_matrix()

#         # 4. 결과 확인
#         explicit_matrix, implicit_matrix = processor.get_matrices()

#         print("\n" + "="*50)
#         print("Explicit User-Item Matrix (상위 5개):")
#         print(explicit_matrix.head())
#         print(f"크기: {explicit_matrix.shape}")
#         print("\n" + "="*50)
#         print("Implicit User-Item Matrix (상위 5개):")
#         print(implicit_matrix.head())
#         print(f"크기: {implicit_matrix.shape}")
#         print("="*50)