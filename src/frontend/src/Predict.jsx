import React from 'react'
import axios from 'axios'
import {
    VStack,
    Button,
    Wrap,
    WrapItem,
    Center,
    Input,
    Box
} from '@chakra-ui/react'

const Endpoint = 'http://localhost:5000/'
const ImageContext = React.createContext(null)
const PredictContext = React.createContext(null)

function Predict() {
    const [image, setImage] = React.useState(null)
    const [prediction, setPrediction] = React.useState(null)
    const handleSubmit = (e) => {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const xmin = formData.get('xmin');
        const xmax = formData.get('xmax');
        const ymin = formData.get('ymin');
        const ymax = formData.get('ymax');
        formData.delete('xmin');
        formData.delete('xmax');
        formData.delete('ymin');
        formData.delete('ymax');
        axios.post(Endpoint + 'predict', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            },
            params: {
                xmin,
                ymin,
                xmax,
                ymax,
            },
            responseType: 'blob'
        }).then(response => {
            setPrediction(response.headers.prediction)
            setImage(response.data)
        })
    }

    return (
        <VStack>
            <form onSubmit={handleSubmit}>
                <Wrap spacing='30px'>
                    <WrapItem>
                        <Input placeholder='xmin' name="xmin" />
                    </WrapItem>
                    <WrapItem>
                        <Input placeholder='ymin' name="ymin" />
                    </WrapItem>
                    <WrapItem>
                        <Input placeholder='xmax' name="xmax" />
                    </WrapItem>
                    <WrapItem>
                        <Input placeholder='ymax' name="ymax" />
                    </WrapItem>
                </Wrap>
                <br />
                <Input type='file' name="file"/>
                <br />
                <br />
                <Center>
                    <Button
                        type='submit'
                        size="md"
                        colorScheme='teal'
                        variant='solid'>
                        Predict
                    </Button>
                </Center>
            </form>
            <br />
            <ImageContext.Provider value={{image}}>
                {image ? <img src={URL.createObjectURL(image)}/> : null }
            </ImageContext.Provider>
            <br />
            <PredictContext.Provider value={{prediction}}>
                {
                    prediction ?
                        <Box bg='teal' borderRadius='md' p={4} color='white'>
                            <Center>
                                <b> Result: {prediction} </b>
                            </Center>
                        </Box>
                    :
                        null
                }
            </PredictContext.Provider>
      </VStack>
    )
}

export default Predict
