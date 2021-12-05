import React from 'react'
import {
    VStack,
    Button,
    Wrap,
    WrapItem,
    Center,
    Input,
    Image,
    Box
} from '@chakra-ui/react'

const axios = require('axios').default
const Endpoint = 'http://localhost:5000/'
const ImageContext = React.createContext(null)

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
        axios.post(Endpoint + 'predict', formData,
                {
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
                }
        ).then(response => {
            setImage(response.data)
        })
    }
      return (
      <VStack>
      <form onSubmit={handleSubmit}>
          <Wrap spacing='30px'>
              <WrapItem>
                <Input placeholder='xmin' name="xmin"  />
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
          <Input type='submit' value='Submit'/>
      </form>

      <ImageContext.Provider value={{image}}>
      {image ? <img src={URL.createObjectURL(image)}/> : null }
      </ImageContext.Provider>
      </VStack>
      )
}

export default Predict