import os

def delete_all_files(folder_path:str=''):

    """
    This function deletes all files in a specified folder.
    :param folder_path: str, The path of the folder whose files you want to delete
    :return None
    """

    files = os.listdir(folder_path)

    if len(files):

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    else:
        return

