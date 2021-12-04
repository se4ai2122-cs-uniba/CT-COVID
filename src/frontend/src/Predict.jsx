import React from 'react'
import {
    VStack,
    Button,
    Wrap,
    WrapItem,
    Center,
    Input
} from '@chakra-ui/react'

const axios = require('axios').default
const Endpoint = 'http://localhost:5000/'

function Predict() {
    const plotPrediction = (response) => {

    }

    const handleSubmit = (e) => {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        const xmin = formData.get('xmin');
        const xmax = formData.get('xmax');
        const ymin = formData.get('ymin');
        const ymax = formData.get('ymax');
        const response = axios.post(Endpoint + 'predict', formData,
            {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                params: {
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                }
            }
        )
        console.log(response.data)
        plotPrediction(response.data)
    }
      return (
      <form onSubmit={handleSubmit}>
        <VStack>
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
          <Input type='submit' value='Submit' />

        </VStack>
      </form>
      )
}

export default Predict
