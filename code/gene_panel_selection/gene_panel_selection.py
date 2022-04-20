from ast import Expr
import pandas


class ExpressionDataset:
    """Encapsulates gene expression data from scRNAseq

    Parameters
    ----------
    expression_data : pandas dataframe
        Dataframe containing count of genes detected per cell. Columns are genes, with gene names contained in the column names.
        Rows are samples (cells), with the sample IDs contained in the index.
    annotation_data : pandas dataframe | None
        Dataframe containing per-cell annotations. Each row is one cell, with sample name contained in the index.
        Columns should include 'cluster', 'subclass', and 'class'.
    """
    def __init__(self, expression_data: pandas.DataFrame, annotation_data=None|pandas.DataFrame):
        self.expression_data = expression_data
        self.annotation_data = annotation_data

    @classmethod
    def load_arrow(cls, gene_file: str, annotation_file: str, expression_file: str):
        """Return dataset loaded from 3 arrow files

        * ``expression_file`` contains one gene per row, one cell per column.
        * ``gene_file`` contains a column 'gene' with the same length and order as rows in expression_file.
        * ``annotation_file`` contains columns 'sample_id', 'cluster', 'subclass', and 'class'. 
        """
        annotations = pandas.read_feather(annotation_file).set_index('sample_id')
        genes = pandas.read_feather(gene_file)
        expression = pandas.read_feather(expression_file)
        expression = expression.set_index(genes['gene']).T

        exp_data = ExpressionDataset(
            expression_data=expression,
            annotation_data=annotations,
        )

        return exp_data


class GenePanelSelection:
    """Contains a gene panel selection as well as information about how the selection was performed.
    """
    def __init__(self, exp_data: ExpressionDataset, gene_panel: list, method: str, args: dict):
        self.exp_data = exp_data
        self.gene_panel = gene_panel
        self.method = method
        self.args = args

    def expression_dataset(self):
        """Return a new ExpressionDataset containing only the genes selected in this panel.
        """
        return ExpressionDataset(
            expression_data=self.exp_data.expression_data[self.gene_panel],
            annotation_data=self.exp_data.annotation_data,
        )


def select_gene_panel(method: str, size: int, data: ExpressionDataset):
    global panel_selection_methods
    assert method in panel_selection_methods, f"Unknown gene panel selection method '{method}'"
    return panel_selection_methods[method](size=size, data=data)


def select_gene_panel_propose(size: int, data: ExpressionDataset):
    raise NotImplementedError()


panel_selection_methods = {
    'PROPOSE': select_gene_panel_propose,
}
