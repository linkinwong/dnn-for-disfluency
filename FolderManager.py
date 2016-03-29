
# coding: utf-8

__author__ = 'linlin'
import os
import logging
import re
import pdb

logger = logging.getLogger(__name__)

class FolderManager(object):
    def __init__(self, root_dir='.'):
        ## 在选定的文件夹里生成更高版本号的文件夹 root_dir - can be relative directory
        # root_dir = '/home/linlin/linlin/word2vec/'
        self.root = os.path.abspath(os.path.dirname( root_dir ))

    def getNewFolderVersionHigher(self,  dir_name):
        ## dir_name - the new folder name you want to creat
        abs_data_directory = self.root
        version_number = 0
        dirs = os.listdir(abs_data_directory)
        for dir in dirs:
            if dir_name in dir and os.path.isdir(dir):
                version_str = re.findall(r'process-\d+',dir)
                print version_str
                if version_str == []:
                    continue
                number_str =''.join((version_str[-1])[8:])
                print "number_str is a number" + number_str
                if True == number_str.isdigit():
                    number= int (number_str)
                    if number>version_number:
                        version_number = number
        new_folder_name = dir_name + "-%d" %(version_number+1)
        folderFullPath = os.path.join(abs_data_directory,new_folder_name )
        os.makedirs(folderFullPath)
        return folderFullPath


    def getAbsFilesPath(self, source_path = '.'):
        # going through all the files in the source_path
        abs_files_path
        for file in os.listdir(source_path):
            abs_path = os.path.realpath( file )
            if os.path.isfile(abs_path):
                abs_files_path.append( abs_path)
        return abs_files_path

    def getAbsFoldersPath(self, source_path = '.'):
        # going through all the sub folders (only one level) in the source_path
        abs_dirs_path=[]
        for dir in os.listdir(source_path):
            abs_dir_path = os.path.realpath( dir )
            if os.path.isdir(abs_dir_path):
                abs_dirs_path.append(abs_dir_path)
        return abs_dirs_path
