
def test_hello(client):
    """Test to check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hi there!"}


def test_predict(client):
    """Test prediction endpoint with valid data"""
    payload = {
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S"
        }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in {0, 1}
    

def test_predict_invalid_data(client):
    """Test prediction input with invalid data"""
    response = client.post("/predict", json={"invalid":"data"})
    assert response.status_code == 422