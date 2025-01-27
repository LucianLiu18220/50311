/**
 *Submitted for verification at Etherscan.io on 2018-07-27
*/

pragma solidity ^0.4.11;

contract StringDump {
    event Event(string value);

    function emitEvent(string value) public {

        Event(value);
    }
}