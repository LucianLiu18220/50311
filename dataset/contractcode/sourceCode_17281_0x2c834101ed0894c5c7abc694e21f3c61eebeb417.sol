/**
 *Submitted for verification at Etherscan.io on 2017-11-09
*/

pragma solidity ^0.4.0;

contract Eventer {
  event Record(
    address _from,
    string _message
  );

  function record(string message) {
    Record(msg.sender, message);
  }
}