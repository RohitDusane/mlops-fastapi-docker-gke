import traceback
import sys
from src.diabetes_risk_prediction.logger.logging import logging

class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message, error_detail)
    
    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):

        _, _, exc_tb = traceback.sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error in {file_name} , line {line_number} : {error_message}"
    
    def __str__(self):
        return self.error_message

# Check the Exception
if __name__ == "__main__":
    try:
        a = 10 / 0  # Division by zero to trigger exception
    except Exception as e:
        logging.info(f"❌ Error occurred: {e}")
        raise CustomException(str(e), sys)