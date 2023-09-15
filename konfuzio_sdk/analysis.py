
from typing import List
import matplotlib.pyplot as plt
import requests

from konfuzio_sdk.data import Project, Category, Document, Label

class APIAccess:
    """
    Provide simplified access to Konfuzio's API endpoint.
    """

    def __init__(self, token: str, endpoint: str = 'https://app.konfuzio.com/api/v3/'):
        """
        Class Constructor.

        :param token: String obtained by a call to the authentication endpoint.
        :param endpoint: API endpoint. Change this only for on-premise setup.
        """
        self.token = token
        self.api_url = endpoint
        self.auth_header = {
            "Authorization": f'Token {self.token}'
        }

    def do_get(self, url):
        """
        Perform a GET request to the specified URL with authorization headers.

        :param url: The URL to send the GET request to.
        :return: JSON response from the GET request, encoded as dict.
        """
        return requests.get(url, headers=self.auth_header).json()

    def do_post(self, url, payload):
        """
        Perform a POST request to the specified URL with JSON payload.

        :param url: The URL to send the POST request to.
        :param payload: The JSON payload to include in the request.
        :return: Response from the POST request.
        """
        return requests.post(url, json=payload)

    def do_patch(self, url, payload):
        """
        Perform a PATCH request to the specified URL with JSON payload.

        :param url: The URL to send the PATCH request to.
        :param payload: The JSON payload to include in the request.
        :return: Response from the PATCH request.
        """
        return requests.patch(url, json=payload)    

    def get_projects(self):
        """
        Retrieve a list of projects from the API.

        :return: List of projects, encoded as dict.
        """
        url = self.api_url + 'projects/'
        return self.do_get(url)

    def get_annotation(self, id):
        """
        Retrieve annotation details for a specific ID.

        :param id: The ID of the annotation to retrieve.
        :return: Details of the annotation, encoded as dict.
        """
        return self.do_get(self.api_url + f'annotations/{str(id)}/')

    def set_annotation(self, id, value):
        """
        Set annotation for a specific ID with the provided value.

        :param id: The ID of the annotation to set.
        :param value: The value to set for the annotation.
        :return: Response from the POST request.
        """
        return self.do_post(self.api_url + f'annotations/{str(id)}/', value)
    
    @staticmethod
    def generate_token(username: str, password: str, auth_endpoint: str = 
                       'https://app.konfuzio.com/api/v3/auth/') -> str:
        """
        Generates an API access token. Call this method only once, then store 
        your token somewhere else.

        :param username: The email address corresponding to your account
        :param password: Your Konfuzio password
        :param auth_endpoint: The authentication API endpoint. Change this only 
        for on-premise setup.
        :returns: The API authentication token as string in a dictonary with 
        key 'token'.
        """

        payload = {
            "username": username,
            "password": password
        }

        response = requests.post(auth_endpoint, json=payload)
        return response.json()

class Analysis:
    """A collection of data analysis tools."""

    def __init__(self, project_id: int) -> None:
        self.project = Project(id_=project_id, update=True)

    def _count_docs_by_label(self, label_name_clean: str, data: List[Document]) -> dict:
        """
        Counts how many documents contain an annotation for a given label, 
        aggregating by unique values of the given label.
        
        Args:
            label_name_clean (str): Name of the label for which the documents 
                should be counted. Note that the 'clean' label name contains an 
                underscore if it's made up of two words, for example 'Company_name'.
            data (List[Document]): A list of Document objects to analyze.

        Returns:
            dict: A dictionary of counts where keys represent unique label values, 
                and values represent the number of documents containing each value.
        """

        counts = {}
        for doc in data:
            annotations = [a.offset_string for a in doc.annotations() if a.label.name_clean == label_name_clean]
            if len(annotations) > 0:
                key = annotations[0][0] # only consider first, assuming same document contains duplicates
                if key in counts:
                    counts[key] = counts[key] + 1
                else:
                    counts[key] = 1
        return counts
    
    def count_train_docs_by_label(self, label_name_clean: str) -> dict:
        """
        Counts how many training documents contain an annotation for a given label, 
        aggregating by unique values of the given label.

        Args:
            label_name_clean (str): Name of the label for which the training documents 
                should be counted. Note that the 'clean' label name contains an 
                underscore if it's made up of two words, for example 'Company_name'.

        Returns:
            dict: A dictionary of counts where keys represent unique label values, 
                and values represent the number of training documents containing each value.
        """        
        return self._count_docs_by_label(label_name_clean, self.project.documents)
    
    def count_test_docs_by_label(self, label_name_clean: str) -> dict:
        """
        Counts how many test documents contain an annotation for a given label, 
        aggregating by unique values of the given label.

        Args:
            label_name_clean (str): Name of the label for which the test documents 
                should be counted. Note that the 'clean' label name contains an 
                underscore if it's made up of two words, for example 'Company_name'.

        Returns:
            dict: A dictionary of counts where keys represent unique label values, 
                and values represent the number of test documents containing each value.
        """        
        return self._count_docs_by_label(label_name_clean, self.project.test_documents)
    

class Plot:
    """
    A collection of various plotting snippets, offered as static methods.
    """
    def __init__(self) -> None:
        pass

    def plot_doc_counts(data: dict, xlabel, title='', sort=True, save_fig=True, file_name='doc_counts.png') -> None:
        """
        Plots a bar chart representing document counts.

        Args:
            data (dict): A dictionary where keys are labels and values are document counts.
            xlabel (str): Label for the x-axis.
            title (str, optional): Title for the plot. Default is an empty string.
            sort (bool, optional): Whether to sort the data by document count. Default is True.
            save_fig (bool, optional): Whether to save the figure. Default is True.
            file_name (str, optional): Name for the saved file. Default is 'doc_counts.png'.

        Returns:
            None: The function displays the plot but does not return any value.
        """
        # Sort the data if sort is True
        if sort:
            data = dict(sorted(data.items(), key=lambda x:x[1]))
        
        # Create a figure with specified size
        plt.figure(figsize=(30,15))
        
        # Rotate x-axis labels for better visibility and set font size
        plt.xticks(rotation=90, fontsize=14)
        
        # Create a bar plot with data
        plt.bar(data.keys(), data.values(), width=0.5)
        
        # Set x-axis label and y-axis label with specified font size
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("Document count", fontsize=14)
        
        # Add a title if provided
        if title:
            plt.title(title, fontsize=14)
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save the figure if save_fig is True
        if save_fig:
            plt.savefig(file_name)
        
        # Display the plot
        plt.show()


    def plot_compare_doc_counts(data1: dict, data2: dict, labels=['Dictionary 1', 'Dictionary 2'], 
                                xlabel='Keys', ylabel='Values', title='', save_fig=True, 
                                file_name='doc_counts_compare') -> None:
        """
        Plots a bar chart comparing document counts from two dictionaries.
        This is typically used if one wants to compare the counts originating 
        from the train vs. the test data.

        Args:
            data1 (dict): The first dictionary containing document counts.
            data2 (dict): The second dictionary containing document counts.
            labels (list, optional): Labels for the two datasets. Default is ['Dictionary 1', 'Dictionary 2'].
            xlabel (str, optional): Label for the x-axis. Default is 'Keys'.
            ylabel (str, optional): Label for the y-axis. Default is 'Values'.
            title (str, optional): Title for the plot. Default is an empty string.
            save_fig (bool, optional): Whether to save the figure. Default is True.
            file_name (str, optional): Name for the saved file. Default is 'doc_counts_compare'.

        Returns:
            None: The function displays the plot but does not return any value.
        """

        # Merge the keys from both dictionaries
        all_keys = set(data1.keys()) | set(data2.keys())

        # Sort by the first dictionary
        all_keys = sorted(all_keys, key=lambda x: data1.get(x, 0))
        
        # Generate positions for bars
        positions = range(len(all_keys))
        
        # Create bar plot
        fig = plt.figure(figsize=(30,15))
        plt.bar(positions, [data1.get(key, 0) for key in all_keys], width=0.4, label=labels[0])
        plt.bar([pos + 0.4 for pos in positions], [data2.get(key, 0) for key in all_keys], width=0.4, label=labels[1])
        
        # Set x-axis labels and positions
        plt.xticks([pos + 0.2 for pos in positions], all_keys, rotation=90)
        
        # Add legend and labels
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)

        plt.tight_layout()
        
        # Show the plot
        plt.show()        

