import json
from pathlib import Path

from flask import Flask, request
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__,
                static_url_path='/models/checkpoint',
                static_folder='checkpoint')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///models.db'
db = SQLAlchemy(app)
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    arch_name = db.Column(db.String, unique=True)
    path_to_weights = db.Column(db.String, unique=False)
    trained_epochs = db.Column(db.Integer, unique=False)
    model_params = db.Column(db.String, unique=False)
    model_results = db.Column(db.String, unique=False)
    state = db.Column(db.String, unique=False)

    def __repr__(self):
        return f'<Model {self.arch_name} - {self.state}>'

    def to_json(self):
        d = self.__dict__
        d = d.copy()
        d.pop('_sa_instance_state', None)
        d['model_params'] =dict(json.loads(self.model_params))
        d['model_results'] =dict(json.loads(self.model_results))
        return d


db.create_all()
db.session.commit()

def model_exists(model: Model) -> bool:
    models = Model.query.all()
    for m in models:
        if m.arch_name == model.arch_name:
            return True
    return False

def add_model_if_not_exists(model: Model):
    if not model_exists(model):
        db.session.add(model)
        db.session.commit()
        print(f'Model added {model}')

def parse_pth_to_model(filename: str, path: str):
    parts = filename.split('.')[0].split('-')
    arch_name = '-'.join(parts)
    epochs = int(parts[3])
    conv = int(parts[1][1:])
    fcl = int(parts[2][1:])
    model_params = {'conv': conv, 'fcl': fcl}
    model_params_string = json.dumps(model_params)
    model = Model(arch_name=arch_name, model_params=model_params_string, model_results='{}', state='waiting', path_to_weights=path, trained_epochs=epochs)
    return model

def list_all_saved_models(base_path: str):
    p = Path(base_path)
    files = p.glob('*/*/*.pth')
    for file in files:
        model = parse_pth_to_model(file.name, str(file))
        print(model)
        add_model_if_not_exists(model)
    # print(files)

def dispatch_model(model: Model):
    model.state = 'queued'
    db.session.commit()

def get_all_models():
    return Model.query.all()

def get_waiting_models():
    return Model.query.filter_by(state='waiting').all()

def next_model():
    base_path = 'checkpoint'
    list_all_saved_models(base_path)
    models = get_waiting_models()
    if len(models):
        model = models[0]
        # Dispatch model
        model_obj = model.to_json()
        dispatch_model(model)
        print(model_obj)
        result = {
            'result': True,
            'model_url': model.path_to_weights,
            'model_params': model.model_params,
            'db_obj': model_obj
        }
    else:

    # for model in models:
    #     print(f'Model: {model}')
    # model_name = 'vgg_c2_f2-1-regular.pth'
    # model_path = f'vgg_c2_f2/Thursday_10_December_2020_16h_02m_34s/{model_name}'
    # # Base case return false
    # if True:
    #     result = {
    #         'result': True,
    #         'model_url': model_path,
    #         'model_params': {
    #             'conv': 2,
    #             'fcl': 2
    #         }
    #     }
    # else:
        result = {
            'result': False
        }
    return result



@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/next')
def next():
    d = next_model()
    return jsonify(d)

@app.route('/save-result', methods=['POST'])
def result():
    req_data = request.get_json()

    # d = next_model()
    return jsonify(req_data)

if __name__ == '__main__':
    # list_all_saved_models('checkpoint')
    app.run(debug=True)