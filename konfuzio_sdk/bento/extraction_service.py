import bentoml
import jsonpickle

extraction_runner = bentoml.picklable_model.get("extraction:latest").to_runner()

svc = bentoml.Service("extraction_svc", runners=[extraction_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def extract(document_json: str):
    document = jsonpickle.loads(document_json)
    result = await extraction_runner.extract.async_run(document)
    return jsonpickle.dumps(result)
