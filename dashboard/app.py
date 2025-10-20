import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import base64
import json
import requests
import re
import pandas as pd
from io import BytesIO

# 创建Dash应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# 变量名称到中文解释的映射字典
variable_to_chinese = {
    "chain_id": "链ID",
    "ddg": "自由能变化",
    "mut_aa": "突变氨基酸",
    "mut_pos": "突变位置",
    "mutcode": "突变代码",
    "ori_aa": "原氨基酸"
}

# 定义应用的布局
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("PDB文件处理和突变输入", className="text-center my-4"),
                width=12
            )
        ),

        dcc.Store(id='stored-results', data=[]),
        dcc.Store(id='uploaded-files', data=[]),
        dcc.Download(id='download-dataframe-xlsx'),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Upload(
                            id='upload-pdb',
                            children=html.Div([
                                html.A('点击选择PDB文件')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=True
                        ),
                        html.Div(id='file-info', className='my-2'),
                        dbc.Button('删除文件', id='delete-button', color='danger', className='mt-2')
                    ],
                    width=6
                ),
                dbc.Col(
                    [
                        dcc.Textarea(
                            id='mutations',
                            placeholder='输入突变信息，每行一个...',
                            style={'width': '100%', 'height': '200px'},
                        ),
                        dbc.Button('删除输出', id='delete-output-button', color='danger', className='mt-2')
                    ],
                    width=6
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button('提交', id='submit-button', color='primary', className='my-4'),
                    width=6,
                    className='text-center'
                ),
                dbc.Col(
                    dcc.Loading(
                        id='loading',
                        type='circle',
                        children=html.Div(id='loading-output'),
                    ),
                    width=6,
                    className='text-center'
                )
            ]
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id='output'),
                width=12
            )
        )
    ],
    fluid=True,
)


@app.callback(
    [Output('file-info', 'children'), Output('uploaded-files', 'data'), Output('upload-pdb', 'contents')],
    [Input('upload-pdb', 'contents'), Input('delete-button', 'n_clicks'),
     Input({'type': 'delete-file', 'index': ALL}, 'n_clicks')],
    [State('upload-pdb', 'filename'), State('uploaded-files', 'data')],
    prevent_initial_call=True
)
def handle_file_info(pdb_contents, delete_clicks, delete_file_clicks, filename, uploaded_files):
    ctx = dash.callback_context

    if not ctx.triggered:
        return '', uploaded_files, None

    prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if prop_id == 'delete-button':
        return '', [], None

    if 'index' in ctx.triggered[0]['prop_id']:
        index = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
        uploaded_files.pop(index)
    elif pdb_contents:
        for content, name in zip(pdb_contents, filename):
            uploaded_files.append({'filename': name, 'content': content})

    file_list = html.Ul([
        html.Li([
            f['filename'],
            dbc.Button('×', id={'type': 'delete-file', 'index': i}, color='link',
                       style={'margin-left': '10px', 'color': 'red'})
        ]) for i, f in enumerate(uploaded_files)
    ])
    return html.Div([
        html.H5('上传的文件:'),
        file_list,
        html.Hr()
    ]), uploaded_files, None

@app.callback(
    [Output('stored-results', 'data', allow_duplicate=True),
     Output('uploaded-files', 'data', allow_duplicate=True),
     Output('mutations', 'value', allow_duplicate=True),
     Output('loading-output', 'children'),
     Output('file-info', 'children', allow_duplicate=True)],
    [Input('submit-button', 'n_clicks')],
    [State('mutations', 'value'), State('uploaded-files', 'data'), State('stored-results', 'data')],
    prevent_initial_call=True
)
def handle_submission(submit_clicks, mutations, uploaded_files, stored_results):
    ctx = dash.callback_context

    if not ctx.triggered:
        return stored_results, uploaded_files, '', '', ''

    if not uploaded_files or not mutations:
        return stored_results, uploaded_files, '', '', ''

    mutations_list = [mut.strip() for mut in mutations.split('\n') if mut.strip()]
    files_to_process = min(len(uploaded_files), len(mutations_list))
    results_generated = -1

    for i in range(files_to_process):
        file = uploaded_files[i]

        content_type, content_string = file['content'].split(',')
        decoded = base64.b64decode(content_string)

        # 将上传的PDB文件和突变信息发送到后端进行处理
        files = {'pdb_file': (file['filename'], decoded)}
        data = {'mutations': mutations_list[i]}
        response = requests.post('http://127.0.0.1:8000/process', files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            # 移除pdb_path字段
            for entry in result:
                if 'pdb_path' in entry:
                    del entry['pdb_path']
            stored_results.append((file['filename'], result))
            results_generated += 1
        else:
            try:
                error_message = response.json().get('error', '处理失败，请检查输入的文件和突变信息。')
            except json.JSONDecodeError:
                error_message = response.text or '处理失败，请检查输入的文件和突变信息。'

            # 提取并清理错误信息，只保留第一个句号之前的内容
            runtime_error_match = re.search(r'RuntimeError:\s*(.*?)(?:\.|$)', error_message)
            if runtime_error_match:
                error_message = runtime_error_match.group(1).strip()
            else:
                error_message = re.sub(r'<[^>]+>', '', error_message).split('.')[0].strip()

            stored_results.append((file['filename'], error_message))
            results_generated += 1

    # 从旧到新移除处理过的文件
    uploaded_files = uploaded_files[results_generated:]

    # 更新文件列表显示
    file_list = html.Ul([
        html.Li([
            f['filename'],
            dbc.Button('×', id={'type': 'delete-file', 'index': i}, color='link',
                       style={'margin-left': '10px', 'color': 'red'})
        ]) for i, f in enumerate(uploaded_files)
    ])

    return stored_results, uploaded_files, '', '', html.Div([
        html.H5('上传的文件:'),
        file_list,
        html.Hr()
    ])


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    [Input({'type': 'download-file', 'index': ALL}, 'n_clicks')],
    [State('stored-results', 'data')],
    prevent_initial_call=True
)
def download_excel(n_clicks, stored_results):
    ctx = dash.callback_context
    if not ctx.triggered or not stored_results:
        return None

    button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    if n_clicks[button_id]:
        filename, result = stored_results[button_id]
        output = BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if isinstance(result, str):
                df = pd.DataFrame([[result]], columns=["错误信息"])
            else:
                df = pd.DataFrame(result)
                df.rename(columns=variable_to_chinese, inplace=True)
            df.to_excel(writer, index=False, sheet_name='Sheet1')

        output.seek(0)
        return dcc.send_bytes(output.getvalue(), f"{filename}.xlsx")


@app.callback(
    [Output('output', 'children')],
    [Input('stored-results', 'data')],
    prevent_initial_call=True
)
def update_output_div(stored_results):
    return [format_results_as_table(stored_results)]


@app.callback(
    [Output('output', 'children', allow_duplicate=True),
     Output('stored-results', 'data')],
    [Input('delete-output-button', 'n_clicks')],
    prevent_initial_call=True
)
def clear_output(n_clicks):
    return '', []


def format_results_as_table(results):
    tables = []
    for index, (filename, result) in enumerate(results):
        # 文件名
        file_header = html.H5(f'文件: {filename}')
        download_button = dbc.Button('下载', id={'type': 'download-file', 'index': index}, color='primary', className='ml-2')
        if isinstance(result, str):
            # 如果是错误信息
            tables.append(html.Div([file_header, html.Div(result), download_button], className="my-4"))
        else:
            # 获取表头，并替换为中文解释
            table_header = [
                html.Thead(html.Tr([html.Th(variable_to_chinese.get(col, col)) for col in result[0].keys()]))
            ]
            # 获取表身
            rows = []
            for entry in result:
                row = html.Tr([html.Td(entry[col]) for col in entry.keys()])
                rows.append(row)
            table_body = [html.Tbody(rows)]
            tables.append(html.Div([file_header,
                                    dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True,
                                              striped=True, className="my-4"), download_button, html.Br()], className="my-4"))
    return html.Div(tables)


if __name__ == "__main__":
    app.run_server(debug=True)
