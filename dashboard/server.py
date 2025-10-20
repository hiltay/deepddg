from flask import Flask, request, jsonify
from flask_cors import CORS
from macroddg.app.launching.main import main, GlobalOptions
import os
import tempfile
import json

app = Flask(__name__)
CORS(app)

@app.route('/process', methods=['POST'])
def process():
    pdb_file = request.files['pdb_file']
    mutations = request.form['mutations']

    with tempfile.TemporaryDirectory() as tmpdirname:
        pdb_path = os.path.join(tmpdirname, pdb_file.filename)
        pdb_file.save(pdb_path)

        input_json = {
            "pdb_path": pdb_path,
            "mutcodes": mutations.split(',')
        }
        input_json_filepath = os.path.join(tmpdirname, 'input.json')
        with open(input_json_filepath, 'w') as f:
            json.dump(input_json, f)

        # 打印输入的数据以便调试
        print(f'pdb_path: {pdb_path}')
        print(f'mutations: {mutations}')
        print(f'output_dir: {tmpdirname}')
        try:
            opts = GlobalOptions(
                input_pdb=pdb_path,
                mutations=mutations,
                output_dir=tmpdirname
            )
        except Exception as e:
            print(f'Error creating GlobalOptions: {e}')
            return jsonify({'error': f'处理失败，输入的数据格式不正确。RuntimeError: {e}'}), 400

        result_code = main(opts)

        if result_code == 0:
            with open(os.path.join(tmpdirname, 'output.json')) as f:
                result = json.load(f)
            return jsonify(result)
        else:
            return jsonify({'error': '处理失败'}), 400

if __name__ == "__main__":
    app.run(debug=True, port=8000)
